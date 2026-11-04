/*
 * The Python Imaging Library
 * $Id$
 *
 * hash tables used by the image quantizer
 *
 * history:
 * 98-09-10 tjs  Contributed
 * 98-12-29 fl   Added to PIL 1.0b1
 *
 * Written by Toby J Sargeant <tjs@longford.cs.monash.edu.au>.
 *
 * Copyright (c) 1998 by Toby J Sargeant
 * Copyright (c) 1998 by Secret Labs AB
 *
 * See the README file for information on usage and redistribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "QuantHash.h"

typedef struct _HashNode {
    struct _HashNode *next;
    uint32_t hash; /* hashFunc(key), kept so chain walks compare integers */
    HashKey_t key;
    HashVal_t value;
} HashNode;

/* Nodes come from blocks owned by the table instead of one malloc each;
   freed nodes are reused. The chains are the same as before. */
#define NODES_PER_BLOCK 4096

typedef struct _HashNodeBlock {
    struct _HashNodeBlock *next;
    HashNode nodes[NODES_PER_BLOCK];
} HashNodeBlock;

struct _HashTable {
    HashNode **table;
    uint32_t length;
    uint32_t count;
    HashFunc hashFunc;
    HashCmpFunc cmpFunc;
    void *userData;
    HashNodeBlock *blocks;
    uint32_t blockUsed;
    HashNode *freeNodes;
    int orderedByHash; /* cmpFunc orders keys as hashFunc's values */
    uint64_t fastmodM; /* for hash % length without a division */
};

/* The bucket of a hash: hash % length, by the multiply-shift of Lemire's
   fastmod where a 128-bit product is available, which gives the same
   value for every 32-bit hash and length. */
static inline void
_hashtable_set_length(HashTable *h, uint32_t length) {
    h->length = length;
    h->fastmodM = UINT64_MAX / length + 1;
}

static inline uint32_t
_hashtable_bucket(const HashTable *h, uint32_t hash) {
#ifdef __SIZEOF_INT128__
    uint64_t lowbits = h->fastmodM * hash;
    return (uint32_t)(((unsigned __int128)lowbits * h->length) >> 64);
#else
    return hash % h->length;
#endif
}

/* Walks key's chain, sorted ascending by cmpFunc: returns the node with
   key, or NULL with *where the link where a new node goes (before the
   first node that sorts after key, or at the end). A table ordered by
   hash compares the hashes kept in the nodes instead of calling
   cmpFunc, which gives the same positions. */
static inline HashNode *
_hashtable_find(const HashTable *h, uint32_t hash, HashKey_t key, HashNode ***where) {
    HashNode **n = &h->table[_hashtable_bucket(h, hash)];
    if (h->orderedByHash) {
        for (; *n; n = &((*n)->next)) {
            uint32_t nh = (*n)->hash;
            if (nh == hash) {
                *where = n;
                return *n;
            } else if (nh > hash) {
                break;
            }
        }
    } else {
        for (; *n; n = &((*n)->next)) {
            int i = h->cmpFunc(h, (*n)->key, key);
            if (!i) {
                *where = n;
                return *n;
            } else if (i > 0) {
                break;
            }
        }
    }
    *where = n;
    return NULL;
}

static HashNode *
_hashtable_node_alloc(HashTable *h) {
    HashNode *node = h->freeNodes;
    if (node) {
        h->freeNodes = node->next;
        return node;
    }
    if (!h->blocks || h->blockUsed == NODES_PER_BLOCK) {
        HashNodeBlock *block = malloc(sizeof(HashNodeBlock));
        if (!block) {
            return NULL;
        }
        block->next = h->blocks;
        h->blocks = block;
        h->blockUsed = 0;
    }
    return &h->blocks->nodes[h->blockUsed++];
}

static void
_hashtable_node_free(HashTable *h, HashNode *node) {
    node->next = h->freeNodes;
    h->freeNodes = node;
}

#define MIN_LENGTH 11
#define RESIZE_FACTOR 3

static int
_hashtable_insert_node(HashTable *, HashNode *, int, int, CollisionFunc);

HashTable *
hashtable_new(HashFunc hf, HashCmpFunc cf) {
    HashTable *h;
    h = malloc(sizeof(HashTable));
    if (!h) {
        return NULL;
    }
    h->hashFunc = hf;
    h->cmpFunc = cf;
    _hashtable_set_length(h, MIN_LENGTH);
    h->count = 0;
    h->userData = NULL;
    h->blocks = NULL;
    h->blockUsed = 0;
    h->freeNodes = NULL;
    h->orderedByHash = 0;
    h->table = malloc(sizeof(HashNode *) * h->length);
    if (!h->table) {
        free(h);
        return NULL;
    }
    memset(h->table, 0, sizeof(HashNode *) * h->length);
    return h;
}

static uint32_t
_findPrime(uint32_t start, int dir) {
    static int unit[] = {0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0};
    uint32_t t;
    while (start > 1) {
        if (!unit[start & 0x0f]) {
            start += dir;
            continue;
        }
        for (t = 2; t < sqrt((double)start); t++) {
            if (!start % t) {
                break;
            }
        }
        if (t >= sqrt((double)start)) {
            break;
        }
        start += dir;
    }
    return start;
}

static void
_hashtable_rehash(HashTable *h, CollisionFunc cf, uint32_t newSize) {
    HashNode **oldTable = h->table;
    uint32_t i;
    HashNode *n, *nn;
    uint32_t oldSize;
    oldSize = h->length;
    h->table = malloc(sizeof(HashNode *) * newSize);
    if (!h->table) {
        h->table = oldTable;
        return;
    }
    _hashtable_set_length(h, newSize);
    h->count = 0;
    memset(h->table, 0, sizeof(HashNode *) * h->length);
    for (i = 0; i < oldSize; i++) {
        for (n = oldTable[i]; n; n = nn) {
            nn = n->next;
            /* the hash function's result may have changed (the pixel
               hash's scale), so each node's is computed again */
            n->hash = h->hashFunc(h, n->key);
            _hashtable_insert_node(h, n, 0, 0, cf);
        }
    }
    free(oldTable);
}

void
hashtable_order_by_hash(HashTable *h) {
    h->orderedByHash = 1;
}

static void
_hashtable_resize(HashTable *h) {
    uint32_t newSize;
    uint32_t oldSize;
    oldSize = h->length;
    newSize = oldSize;
    if (h->count * RESIZE_FACTOR < h->length) {
        newSize = _findPrime(h->length / 2 - 1, -1);
    } else if (h->length * RESIZE_FACTOR < h->count) {
        newSize = _findPrime(h->length * 2 + 1, +1);
    }
    if (newSize < MIN_LENGTH) {
        newSize = oldSize;
    }
    if (newSize != oldSize) {
        _hashtable_rehash(h, NULL, newSize);
    }
}

static int
_hashtable_insert_node(
    HashTable *h, HashNode *node, int resize, int update, CollisionFunc cf
) {
    HashNode **n, *nv;

    /* node->hash is set by the caller */
    nv = _hashtable_find(h, node->hash, node->key, &n);
    if (nv) {
        if (cf) {
            nv->key = node->key;
            cf(h, &(nv->key), &(nv->value), node->key, node->value);
            _hashtable_node_free(h, node);
            return 1;
        } else {
            nv->key = node->key;
            nv->value = node->value;
            _hashtable_node_free(h, node);
            return 1;
        }
    }
    if (!update) {
        node->next = *n;
        *n = node;
        h->count++;
        if (resize) {
            _hashtable_resize(h);
        }
        return 1;
    } else {
        return 0;
    }
}

static int
_hashtable_insert(HashTable *h, HashKey_t key, HashVal_t val, int resize, int update) {
    HashNode **n, *nv;
    HashNode *t;
    uint32_t hash = h->hashFunc(h, key);

    nv = _hashtable_find(h, hash, key, &n);
    if (nv) {
        nv->value = val;
        return 1;
    }
    if (!update) {
        t = _hashtable_node_alloc(h);
        if (!t) {
            return 0;
        }
        t->next = *n;
        *n = t;
        t->hash = hash;
        t->key = key;
        t->value = val;
        h->count++;
        if (resize) {
            _hashtable_resize(h);
        }
        return 1;
    } else {
        return 0;
    }
}

int
hashtable_insert_or_update_computed(
    HashTable *h, HashKey_t key, ComputeFunc newFunc, ComputeFunc existsFunc
) {
    HashNode **n, *nv;
    HashNode *t;
    uint32_t hash = h->hashFunc(h, key);

    nv = _hashtable_find(h, hash, key, &n);
    if (nv) {
        if (existsFunc) {
            existsFunc(h, nv->key, &(nv->value));
        } else {
            return 0;
        }
        return 1;
    }
    t = _hashtable_node_alloc(h);
    if (!t) {
        return 0;
    }
    t->hash = hash;
    t->key = key;
    t->next = *n;
    *n = t;
    if (newFunc) {
        newFunc(h, t->key, &(t->value));
    } else {
        *n = t->next;
        _hashtable_node_free(h, t);
        return 0;
    }
    h->count++;
    _hashtable_resize(h);
    return 1;
}

int
hashtable_insert(HashTable *h, HashKey_t key, HashVal_t val) {
    return _hashtable_insert(h, key, val, 1, 0);
}

void
hashtable_foreach_update(HashTable *h, IteratorUpdateFunc i, void *u) {
    HashNode *n;
    uint32_t x;

    if (h->table) {
        for (x = 0; x < h->length; x++) {
            for (n = h->table[x]; n; n = n->next) {
                i(h, n->key, &(n->value), u);
            }
        }
    }
}

void
hashtable_foreach(HashTable *h, IteratorFunc i, void *u) {
    HashNode *n;
    uint32_t x;

    if (h->table) {
        for (x = 0; x < h->length; x++) {
            for (n = h->table[x]; n; n = n->next) {
                i(h, n->key, n->value, u);
            }
        }
    }
}

void
hashtable_free(HashTable *h) {
    HashNodeBlock *block, *next;

    for (block = h->blocks; block; block = next) {
        next = block->next;
        free(block);
    }
    free(h->table);
    free(h);
}

void
hashtable_rehash_compute(HashTable *h, CollisionFunc cf) {
    _hashtable_rehash(h, cf, h->length);
}

int
hashtable_lookup(const HashTable *h, const HashKey_t key, HashVal_t *valp) {
    HashNode **where, *n;

    n = _hashtable_find(h, h->hashFunc(h, key), key, &where);
    if (n) {
        *valp = n->value;
        return 1;
    }
    return 0;
}

uint32_t
hashtable_get_count(const HashTable *h) {
    return h->count;
}

void *
hashtable_get_user_data(const HashTable *h) {
    return h->userData;
}

void *
hashtable_set_user_data(HashTable *h, void *data) {
    void *r = h->userData;
    h->userData = data;
    return r;
}

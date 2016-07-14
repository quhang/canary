// Written by Christian Bienia
// The code in this file implements a memory pool of Cell structures.
// It serves three purposes:
//   1.) To minimize calls to malloc and free as much as possible
//   2.) To reuse cell structures as much as possible
//   3.) To eliminate unnecessary synchronization for memory allocation

#include <iostream>

#include <stdlib.h>
#include <assert.h>

#include "fluid.h"
#include "cellpool.h"

/* *** REMINDER ***
 The following asserts were added to the serial program:
 1. assert struct Cell aligned
 2. assert struct Cell and struct Cell_aux same size
*/

// Allocate and initialize a new data block for `cells' number of cells
// Data blocks will have the following format:
//
//   | struct datablockhdr | struct Cell | struct Cell | ..... |
//
// The cells inside the block will be connected to a NULL-terminated linked list
// with the cell at the lowest memory location being the first of its elements.
static struct datablockhdr *cellpool_allocblock(int cells) {
  struct datablockhdr *block = NULL;
  struct Cell *temp1, *temp2;
  int i;

  // allocate a full block
  assert(cells > 0);
  int rv =
      posix_memalign((void **)(&block), CACHELINE_SIZE,
                     sizeof(struct datablockhdr) + cells * sizeof(struct Cell));
  if (rv != 0) {
    exit(1);
  }

  // initialize header and cells
  block->next = NULL;
  temp1 = (struct Cell *)(block + 1);
  for (i = 0; i < cells; i++) {
    // If all structures are correctly padded then all pointers should also be
    // correctly aligned,
    // but let's verify that nevertheless because the padding might change.
    assert((uint64_t)(temp1) % sizeof(void *) == 0);
    if (i != cells - 1) {
      temp2 = temp1 + 1;
      temp1->next = temp2;
      temp1 = temp2;
    } else {
      // last Cell structure in block
      temp1->next = NULL;
    }
  }

  return block;
}

// Initialize the memory pool
// particles is used to determine the initial capacity and should correspond to
// the
// number of particles that the pool is expected to manage
void cellpool_init(cellpool *pool, int particles) {
  int ALLOC_MIN_CELLS = 1024;
  assert(sizeof(struct datablockhdr) % CACHELINE_SIZE == 0);
  assert(pool != NULL);
  assert(particles > 0);

  // Allocate the initial data, let's start with 4 times more cells than
  // best case (ignoring statically allocated Cells structures)
  pool->alloc =
      4 *
      (particles /
       PARTICLES_PER_CELL);  // PARTICLES_PER_CELL particles per cell structure
  pool->alloc = pool->alloc < ALLOC_MIN_CELLS ? ALLOC_MIN_CELLS : pool->alloc;
  pool->datablocks = cellpool_allocblock(pool->alloc);
  pool->cells = (struct Cell *)(pool->datablocks + 1);
}

// Get a Cell structure from the memory pool
Cell *cellpool_getcell(cellpool *pool) {
  struct Cell *temp;

  assert(pool != NULL);

  // If no more cells available then allocate more
  if (pool->cells == NULL) {
    // keep doubling the number of cells
    struct datablockhdr *block = cellpool_allocblock(pool->alloc);
    pool->alloc = 2 * pool->alloc;
    block->next = pool->datablocks;
    pool->datablocks = block;
    pool->cells = (struct Cell *)(pool->datablocks + 1);
  }

  // return first cell in list
  temp = pool->cells;
  pool->cells = temp->next;
  temp->next = NULL;
  return temp;
}

// Return a Cell structure to the memory pool
void cellpool_returncell(cellpool *pool, Cell *cell) {
  assert(pool != NULL);
  assert(cell != NULL);
  cell->next = pool->cells;
  pool->cells = cell;
}

// Destroy the memory pool
void cellpool_destroy(cellpool *pool) {
  assert(pool != NULL);

  // iterate through data blocks and free them all, this will also free all
  // cells
  struct datablockhdr *ptr = pool->datablocks;
  struct datablockhdr *temp;
  while (ptr != NULL) {
    temp = ptr;
    ptr = ptr->next;
    free(temp);
  }
}

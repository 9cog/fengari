# Tensor Logic for Fengari

This document describes the Tensor Logic implementation in Fengari, which bridges neural networks and symbolic reasoning using tensor operations.

## Overview

Tensor Logic is a framework that unifies neural and symbolic AI paradigms by representing logical predicates and rules as tensor operations. This implementation is based on the paper ["Tensor Logic: The Language of AI" by Pedro Domingos](https://arxiv.org/abs/2510.12269).

### Key Features

- **Unified Framework**: Combines neural, symbolic, and statistical AI in one language
- **Dual Modes**: Supports both Boolean (exact logic) and continuous (probabilistic learning) modes
- **Scalability**: Leverages tensor operations for efficient computation
- **Reasoning in Embedding Space**: Enables symbolic reasoning directly in continuous vector spaces

## Tensor Library

The `tensor` library provides basic tensor operations.

### Creating Tensors

```lua
-- Create a 2x3 tensor
local t = tensor.new({2, 3})

-- Get tensor shape
local shape = tensor.shape(t)
print(shape[1], shape[2])  -- Output: 2 3
```

### Setting and Getting Values

```lua
local t = tensor.new({2, 2})

-- Set values (1-indexed in Lua)
tensor.set(t, 1, 1, 5.0)
tensor.set(t, 1, 2, 3.0)

-- Get values
local val = tensor.get(t, 1, 1)  -- Returns 5.0
```

### Tensor Operations

#### Matrix Multiplication

```lua
local t1 = tensor.new({2, 3})
local t2 = tensor.new({3, 2})

local result = tensor.matmul(t1, t2)  -- 2x2 result
```

#### Activation Functions

```lua
-- Sigmoid
local t = tensor.new({3})
tensor.set(t, 1, 0)
local activated = tensor.sigmoid(t)

-- ReLU
local activated2 = tensor.relu(t)
```

### Boolean Mode Conversion

```lua
local t = tensor.new({4})
tensor.set(t, 1, 0.2)
tensor.set(t, 2, 0.6)
tensor.set(t, 3, 0.8)
tensor.set(t, 4, 0.3)

-- Convert to boolean (threshold at 0.5 by default)
local bool_t = tensor.bool(t)
-- Result: [0, 1, 1, 0]
```

## Tensor Logic Library

The `tensorlogic` library provides logical reasoning operations using tensors.

### Relations

```lua
-- Create a relation from a list of (from, to, value) tuples
local parent_relations = {
    {1, 2, 1.0},  -- Entity 1 is parent of entity 2
    {2, 3, 1.0},  -- Entity 2 is parent of entity 3
}

local parent = tensorlogic.relation(parent_relations, 3)
```

### Logical Composition

```lua
-- Given parent relation, compute grandparent
local grandparent = tensorlogic.compose(parent, parent)

-- This implements: grandparent(X, Z) ← parent(X, Y) ∧ parent(Y, Z)
```

### Embeddings and Similarity

```lua
-- Create embeddings for 5 entities in 10-dimensional space
local embeddings = tensorlogic.embed(5, 10)

-- Compute cosine similarity
local sim = tensorlogic.similarity(v1, v2)
```

## Examples

### Example 1: Family Relationships

```lua
-- Define parent relationships
local parent_relations = {
    {1, 2, 1.0},  -- Alice is parent of Bob
    {2, 3, 1.0},  -- Bob is parent of Charlie
    {3, 4, 1.0},  -- Charlie is parent of Diana
}

local parent = tensorlogic.relation(parent_relations, 4)

-- Compute grandparent
local grandparent = tensorlogic.compose(parent, parent)

-- Compute great-grandparent
local greatgrandparent = tensorlogic.compose(grandparent, parent)
```

## References

- Domingos, P. (2024). "Tensor Logic: The Language of AI". arXiv:2510.12269
- [Tensor Logic Website](https://tensor-logic.org/)
- [Ben Goertzel's article on Tensor Logic](https://bengoertzel.substack.com/p/tensor-logic-for-bridging-neural)

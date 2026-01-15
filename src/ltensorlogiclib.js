/*jshint esversion: 6 */
"use strict";

const defs    = require('./defs.js');
const lapi    = require('./lapi.js');
const lauxlib = require('./lauxlib.js');
const lua     = require('./lua.js');
const { to_luastring, to_jsstring } = require('./fengaricore.js');
const CT      = defs.constant_types;
const ltensorlib = require('./ltensorlib.js');

/*
** Tensor Logic Library for Fengari
** Implements symbolic reasoning using tensor operations
** Based on "Tensor Logic: The Language of AI" by Pedro Domingos
** 
** This library bridges neural networks and symbolic logic by:
** 1. Representing logical predicates as tensors
** 2. Implementing logical rules as tensor operations
** 3. Supporting both Boolean (exact) and continuous (probabilistic) modes
*/

// Helper functions to get objects from userdata
const gettensor = function(L, idx) {
    const udata = lua.lua_touserdata(L, idx);
    if (!udata || !udata.tensor) {
        lauxlib.luaL_error(L, to_luastring("expected tensor"));
        return null;
    }
    return udata.tensor;
};

const getkb = function(L, idx) {
    const udata = lua.lua_touserdata(L, idx);
    if (!udata || !udata.kb) {
        lauxlib.luaL_error(L, to_luastring("expected knowledge base"));
        return null;
    }
    return udata.kb;
};

// Knowledge base to store predicates
class KnowledgeBase {
    constructor() {
        this.predicates = new Map(); // predicate_name -> tensor
        this.mode = 'boolean'; // 'boolean' or 'continuous'
    }

    addPredicate(name, tensor) {
        this.predicates.set(name, tensor);
    }

    getPredicate(name) {
        return this.predicates.get(name);
    }

    setMode(mode) {
        this.mode = mode;
    }

    getMode() {
        return this.mode;
    }
}

// Create a new knowledge base
const kb_new = function(L) {
    const kb = new KnowledgeBase();
    const udata = lua.lua_newuserdata(L, 0);
    udata.kb = kb;
    lauxlib.luaL_setmetatable(L, to_luastring("knowledgebase"));
    return 1;
};

// Add predicate to knowledge base
const kb_add_predicate = function(L) {

    const kb = getkb(L, 1);
    
    if (!kb) {
        return lauxlib.luaL_error(L, to_luastring("expected knowledge base"));
    }
    
    const name = lauxlib.luaL_checkstring(L, 2);
    
    const tensor = gettensor(L, 3);
    
    kb.addPredicate(to_jsstring(name), tensor);
    return 0;
};

// Get predicate from knowledge base
const kb_get_predicate = function(L) {
    const udata = lua.lua_touserdata(L, 1);
    const kb = getkb(L, 1);
    
    if (!kb) {
        return lauxlib.luaL_error(L, to_luastring("expected knowledge base"));
    }
    
    const name = lauxlib.luaL_checkstring(L, 2);
    const tensor = kb.getPredicate(to_jsstring(name));
    
    if (!tensor) {
        lua.lua_pushnil(L);
        return 1;
    }
    
    const tensorUdata = lua.lua_newuserdata(L, 0);
    tensorUdata.tensor = tensor;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Set reasoning mode (boolean or continuous)
const kb_set_mode = function(L) {
    const udata = lua.lua_touserdata(L, 1);
    const kb = getkb(L, 1);
    
    if (!kb) {
        return lauxlib.luaL_error(L, to_luastring("expected knowledge base"));
    }
    
    const mode = lauxlib.luaL_checkstring(L, 2);
    const modeStr = to_jsstring(mode);
    
    if (modeStr !== 'boolean' && modeStr !== 'continuous') {
        return lauxlib.luaL_error(L, to_luastring("mode must be 'boolean' or 'continuous'"));
    }
    
    kb.setMode(modeStr);
    return 0;
};

// Get reasoning mode
const kb_get_mode = function(L) {
    const udata = lua.lua_touserdata(L, 1);
    const kb = getkb(L, 1);
    
    if (!kb) {
        return lauxlib.luaL_error(L, to_luastring("expected knowledge base"));
    }
    
    lua.lua_pushstring(L, to_luastring(kb.getMode()));
    return 1;
};

// Logical AND operation on predicates
const logic_and = function(L) {
    const ptr1 = lua.lua_touserdata(L, 1);
    const ptr2 = lua.lua_touserdata(L, 2);
    const t1 = gettensor(L, 1);
    const t2 = gettensor(L, 2);
    
    if (!t1 || !t2) return 0;
    
    if (t1.shape.join(',') !== t2.shape.join(',')) {
        return lauxlib.luaL_error(L, to_luastring("tensors must have same shape"));
    }
    
    const result = t1.clone();
    for (let i = 0; i < result.size; i++) {
        // Use minimum as it works for both Boolean and continuous modes
        result.data[i] = Math.min(t1.data[i], t2.data[i]);
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Logical OR operation on predicates
const logic_or = function(L) {
    const t1 = gettensor(L, 1);
    const t2 = gettensor(L, 2);
    
    if (!t1 || !t2) return 0;
    
    if (t1.shape.join(',') !== t2.shape.join(',')) {
        return lauxlib.luaL_error(L, to_luastring("tensors must have same shape"));
    }
    
    const result = t1.clone();
    for (let i = 0; i < result.size; i++) {
        // Use maximum as it works for both Boolean and continuous modes
        result.data[i] = Math.max(t1.data[i], t2.data[i]);
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Logical NOT operation on predicates
const logic_not = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    const result = tensor.clone();
    for (let i = 0; i < result.size; i++) {
        result.data[i] = 1 - result.data[i];
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Compose predicates: implements transitive rules
// Example: grandparent(X, Z) = parent(X, Y) ∧ parent(Y, Z)
// This is represented as matrix multiplication in tensor space
const logic_compose = function(L) {
    const ptr1 = lua.lua_touserdata(L, 1);
    const ptr2 = lua.lua_touserdata(L, 2);
    const t1 = gettensor(L, 1);
    const t2 = gettensor(L, 2);
    
    if (!t1 || !t2) {
        return lauxlib.luaL_error(L, to_luastring("expected tensors"));
    }
    
    if (t1.ndim !== 2 || t2.ndim !== 2) {
        return lauxlib.luaL_error(L, to_luastring("compose requires 2D tensors (relations)"));
    }
    
    if (t1.shape[1] !== t2.shape[0]) {
        return lauxlib.luaL_error(L, to_luastring("incompatible shapes for composition"));
    }
    
    // Matrix multiplication for relational composition
    const result = new ltensorlib.Tensor([t1.shape[0], t2.shape[1]]);
    
    for (let i = 0; i < t1.shape[0]; i++) {
        for (let j = 0; j < t2.shape[1]; j++) {
            let sum = 0;
            for (let k = 0; k < t1.shape[1]; k++) {
                // Take max of the min (logical AND across the chain)
                const val = Math.min(t1.get([i, k]), t2.get([k, j]));
                sum = Math.max(sum, val);
            }
            result.set([i, j], sum);
        }
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Query: apply a rule and return results above threshold
const logic_query = function(L) {
    const udata = lua.lua_touserdata(L, 1);
    const tensor = ptr.tensordata;
    
    if (!tensor) {
        return lauxlib.luaL_error(L, to_luastring("expected tensor"));
    }
    
    const threshold = lauxlib.luaL_optnumber(L, 2, 0.5);
    
    // Return table of indices where value > threshold
    lua.lua_createtable(L, 0, 0);
    let count = 0;
    
    if (tensor.ndim === 1) {
        for (let i = 0; i < tensor.shape[0]; i++) {
            if (tensor.data[i] > threshold) {
                lua.lua_pushinteger(L, i + 1); // Lua is 1-indexed
                lua.lua_pushnumber(L, tensor.data[i]);
                lua.lua_settable(L, -3);
                count++;
            }
        }
    } else if (tensor.ndim === 2) {
        for (let i = 0; i < tensor.shape[0]; i++) {
            for (let j = 0; j < tensor.shape[1]; j++) {
                const val = tensor.get([i, j]);
                if (val > threshold) {
                    lua.lua_createtable(L, 2, 0);
                    lua.lua_pushinteger(L, i + 1);
                    lua.lua_rawseti(L, -2, 1);
                    lua.lua_pushinteger(L, j + 1);
                    lua.lua_rawseti(L, -2, 2);
                    
                    lua.lua_pushnumber(L, val);
                    lua.lua_settable(L, -3);
                    count++;
                }
            }
        }
    }
    
    return 1;
};

// Inference: forward chaining using multiple rules
const logic_infer = function(L) {
    const kbPtr = lua.lua_touserdata(L, 1);
    const kb = kbPtr.kbdata;
    
    if (!kb) {
        return lauxlib.luaL_error(L, to_luastring("expected knowledge base"));
    }
    
    // Get rule as a string
    const rule = lauxlib.luaL_checkstring(L, 2);
    const ruleStr = to_jsstring(rule);
    
    // Simple rule parser: "result = pred1 op pred2"
    // This is a simplified implementation
    const parts = ruleStr.split('=');
    if (parts.length !== 2) {
        return lauxlib.luaL_error(L, to_luastring("invalid rule format"));
    }
    
    const resultName = parts[0].trim();
    const expression = parts[1].trim();
    
    // Parse and evaluate expression (simplified)
    // In a full implementation, this would be a proper parser
    lua.lua_pushstring(L, to_luastring("inference not fully implemented"));
    return lauxlib.luaL_error(L, -1);
};

// Create relation tensor from table
const logic_relation = function(L) {
    lauxlib.luaL_checktype(L, 1, CT.LUA_TTABLE);
    
    // Get dimensions
    const entities = lauxlib.luaL_optinteger(L, 2, 0);
    
    if (entities === 0) {
        return lauxlib.luaL_error(L, to_luastring("number of entities required"));
    }
    
    // Create relation tensor
    const tensor = new ltensorlib.Tensor([entities, entities]);
    
    // Fill from table: table is list of {from, to, value}
    lua.lua_pushnil(L);
    while (lua.lua_next(L, 1)) {
        if (lua.lua_istable(L, -1)) {
            lua.lua_rawgeti(L, -1, 1);
            const from = lauxlib.luaL_checkinteger(L, -1) - 1;
            lua.lua_pop(L, 1);
            
            lua.lua_rawgeti(L, -1, 2);
            const to = lauxlib.luaL_checkinteger(L, -1) - 1;
            lua.lua_pop(L, 1);
            
            lua.lua_rawgeti(L, -1, 3);
            const value = lauxlib.luaL_optnumber(L, -1, 1.0);
            lua.lua_pop(L, 1);
            
            tensor.set([from, to], value);
        }
        lua.lua_pop(L, 1);
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = tensor;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Embedding operations: project entities into continuous space
const logic_embed = function(L) {
    const entities = lauxlib.luaL_checkinteger(L, 1);
    const dim = lauxlib.luaL_checkinteger(L, 2);
    
    // Create embedding tensor
    const tensor = new ltensorlib.Tensor([entities, dim]);
    
    // Random initialization
    for (let i = 0; i < tensor.size; i++) {
        tensor.data[i] = (Math.random() - 0.5) * 2;
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = tensor;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Similarity computation in embedding space
const logic_similarity = function(L) {
    const ptr1 = lua.lua_touserdata(L, 1);
    const ptr2 = lua.lua_touserdata(L, 2);
    const t1 = gettensor(L, 1);
    const t2 = gettensor(L, 2);
    
    if (!t1 || !t2) {
        return lauxlib.luaL_error(L, to_luastring("expected tensors"));
    }
    
    if (t1.ndim !== 1 || t2.ndim !== 1) {
        return lauxlib.luaL_error(L, to_luastring("similarity requires 1D tensors"));
    }
    
    if (t1.size !== t2.size) {
        return lauxlib.luaL_error(L, to_luastring("vectors must have same size"));
    }
    
    // Cosine similarity
    let dot = 0, norm1 = 0, norm2 = 0;
    for (let i = 0; i < t1.size; i++) {
        dot += t1.data[i] * t2.data[i];
        norm1 += t1.data[i] * t1.data[i];
        norm2 += t2.data[i] * t2.data[i];
    }
    
    const similarity = dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
    lua.lua_pushnumber(L, similarity);
    
    return 1;
};

const tensorlogiclib = {
    // Knowledge base operations
    "kb_new":          kb_new,
    "kb_add":          kb_add_predicate,
    "kb_get":          kb_get_predicate,
    "kb_mode":         kb_set_mode,
    "kb_get_mode":     kb_get_mode,
    
    // Logical operations
    "land":            logic_and,
    "lor":             logic_or,
    "lnot":            logic_not,
    "compose":         logic_compose,
    "query":           logic_query,
    "infer":           logic_infer,
    
    // Relation and embedding operations
    "relation":        logic_relation,
    "embed":           logic_embed,
    "similarity":      logic_similarity,
};

const luaopen_tensorlogic = function(L) {
    lauxlib.luaL_newlib(L, tensorlogiclib);
    
    // Create metatable for knowledge base
    lauxlib.luaL_newmetatable(L, to_luastring("knowledgebase"));
    
    lua.lua_pushstring(L, to_luastring("__index"));
    lua.lua_pushvalue(L, -3); // Push the library table
    lua.lua_settable(L, -3);
    
    lua.lua_pop(L, 1);
    
    return 1;
};

module.exports.luaopen_tensorlogic = luaopen_tensorlogic;
module.exports.KnowledgeBase = KnowledgeBase;

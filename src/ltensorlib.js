/*jshint esversion: 6 */
"use strict";

const assert  = require('assert');

const defs    = require('./defs.js');
const lapi    = require('./lapi.js');
const lauxlib = require('./lauxlib.js');
const lua     = require('./lua.js');
const { to_luastring, to_jsstring } = require('./fengaricore.js');
const CT      = defs.constant_types;

/*
** Tensor Logic Library for Fengari
** Implements tensor operations for bridging neural networks and symbolic reasoning
** Based on "Tensor Logic: The Language of AI" by Pedro Domingos
*/

// Tensor structure representation
class Tensor {
    constructor(shape, data = null) {
        this.shape = Array.isArray(shape) ? shape : [shape];
        this.size = this.shape.reduce((a, b) => a * b, 1);
        this.data = data || new Array(this.size).fill(0);
        this.ndim = this.shape.length;
    }

    get(indices) {
        const idx = this._computeIndex(indices);
        return this.data[idx];
    }

    set(indices, value) {
        const idx = this._computeIndex(indices);
        this.data[idx] = value;
    }

    _computeIndex(indices) {
        if (!Array.isArray(indices)) indices = [indices];
        let idx = 0;
        let multiplier = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            idx += indices[i] * multiplier;
            multiplier *= this.shape[i];
        }
        return idx;
    }

    clone() {
        return new Tensor(this.shape.slice(), this.data.slice());
    }
}

// Create a new tensor
const tensor_new = function(L) {
    // Get shape from Lua table
    lauxlib.luaL_checktype(L, 1, CT.LUA_TTABLE);
    
    let shape = [];
    lua.lua_pushnil(L);
    while (lua.lua_next(L, 1)) {
        shape.push(lauxlib.luaL_checkinteger(L, -1));
        lua.lua_pop(L, 1);
    }
    
    if (shape.length === 0) {
        return lauxlib.luaL_error(L, to_luastring("shape must have at least one dimension"));
    }
    
    const tensor = new Tensor(shape);
    
    // Push tensor as userdata
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = tensor;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Get tensor from userdata
const gettensor = function(L, idx) {
    const udata = lua.lua_touserdata(L, idx);
    if (!udata || !udata.tensor) {
        lauxlib.luaL_error(L, to_luastring("expected tensor"));
        return null;
    }
    return udata.tensor;
};

// Get tensor value at indices
const tensor_get = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    const indices = [];
    const nargs = lua.lua_gettop(L);
    for (let i = 2; i <= nargs; i++) {
        indices.push(lauxlib.luaL_checkinteger(L, i) - 1); // Lua is 1-indexed
    }
    
    const value = tensor.get(indices);
    lua.lua_pushnumber(L, value);
    return 1;
};

// Set tensor value at indices
const tensor_set = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    const nargs = lua.lua_gettop(L);
    const value = lauxlib.luaL_checknumber(L, nargs);
    
    const indices = [];
    for (let i = 2; i < nargs; i++) {
        indices.push(lauxlib.luaL_checkinteger(L, i) - 1);
    }
    
    tensor.set(indices, value);
    return 0;
};

// Matrix multiplication (einsum for 2D tensors)
const tensor_matmul = function(L) {
    const t1 = gettensor(L, 1);
    const t2 = gettensor(L, 2);
    
    if (!t1 || !t2) return 0;
    
    if (t1.ndim !== 2 || t2.ndim !== 2) {
        return lauxlib.luaL_error(L, to_luastring("matmul requires 2D tensors"));
    }
    
    if (t1.shape[1] !== t2.shape[0]) {
        return lauxlib.luaL_error(L, to_luastring("incompatible shapes for matmul"));
    }
    
    const result = new Tensor([t1.shape[0], t2.shape[1]]);
    
    for (let i = 0; i < t1.shape[0]; i++) {
        for (let j = 0; j < t2.shape[1]; j++) {
            let sum = 0;
            for (let k = 0; k < t1.shape[1]; k++) {
                sum += t1.get([i, k]) * t2.get([k, j]);
            }
            result.set([i, j], sum);
        }
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Element-wise operations
const tensor_add = function(L) {
    const t1 = gettensor(L, 1);
    const t2 = gettensor(L, 2);
    
    if (!t1 || !t2) return 0;
    
    if (t1.shape.join(',') !== t2.shape.join(',')) {
        return lauxlib.luaL_error(L, to_luastring("tensors must have same shape"));
    }
    
    const result = t1.clone();
    for (let i = 0; i < result.size; i++) {
        result.data[i] = t1.data[i] + t2.data[i];
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Element-wise multiplication
const tensor_mul = function(L) {
    const t1 = gettensor(L, 1);
    const t2 = gettensor(L, 2);
    
    if (!t1 || !t2) return 0;
    
    if (t1.shape.join(',') !== t2.shape.join(',')) {
        return lauxlib.luaL_error(L, to_luastring("tensors must have same shape"));
    }
    
    const result = t1.clone();
    for (let i = 0; i < result.size; i++) {
        result.data[i] = t1.data[i] * t2.data[i];
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Apply sigmoid activation (for continuous mode)
const tensor_sigmoid = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    const result = tensor.clone();
    for (let i = 0; i < result.size; i++) {
        result.data[i] = 1 / (1 + Math.exp(-result.data[i]));
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Apply ReLU activation
const tensor_relu = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    const result = tensor.clone();
    for (let i = 0; i < result.size; i++) {
        result.data[i] = Math.max(0, result.data[i]);
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Convert to Boolean mode (threshold at 0.5)
const tensor_bool = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    const threshold = lauxlib.luaL_optnumber(L, 2, 0.5);
    
    const result = tensor.clone();
    for (let i = 0; i < result.size; i++) {
        result.data[i] = result.data[i] > threshold ? 1 : 0;
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Get tensor shape
const tensor_shape = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    lua.lua_createtable(L, tensor.shape.length, 0);
    for (let i = 0; i < tensor.shape.length; i++) {
        lua.lua_pushinteger(L, tensor.shape[i]);
        lua.lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
};

// Transpose 2D tensor
const tensor_transpose = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    if (tensor.ndim !== 2) {
        return lauxlib.luaL_error(L, to_luastring("transpose requires 2D tensor"));
    }
    
    const result = new Tensor([tensor.shape[1], tensor.shape[0]]);
    for (let i = 0; i < tensor.shape[0]; i++) {
        for (let j = 0; j < tensor.shape[1]; j++) {
            result.set([j, i], tensor.get([i, j]));
        }
    }
    
    const udata = lua.lua_newuserdata(L, 0);
    udata.tensor = result;
    lauxlib.luaL_setmetatable(L, to_luastring("tensor"));
    
    return 1;
};

// Tensor to string representation
const tensor_tostring = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    let str = `Tensor(shape=[${tensor.shape.join(', ')}], data=[`;
    const maxShow = Math.min(tensor.size, 10);
    for (let i = 0; i < maxShow; i++) {
        str += tensor.data[i].toFixed(4);
        if (i < maxShow - 1) str += ', ';
    }
    if (tensor.size > maxShow) {
        str += '...';
    }
    str += '])';
    
    lua.lua_pushstring(L, to_luastring(str));
    return 1;
};

// Fill tensor with value
const tensor_fill = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    const value = lauxlib.luaL_checknumber(L, 2);
    
    for (let i = 0; i < tensor.size; i++) {
        tensor.data[i] = value;
    }
    
    return 0;
};

// Random initialization
const tensor_randn = function(L) {
    const tensor = gettensor(L, 1);
    if (!tensor) return 0;
    
    // Box-Muller transform for normal distribution
    for (let i = 0; i < tensor.size; i += 2) {
        const u1 = Math.random();
        const u2 = Math.random();
        const r = Math.sqrt(-2 * Math.log(u1));
        const theta = 2 * Math.PI * u2;
        tensor.data[i] = r * Math.cos(theta);
        if (i + 1 < tensor.size) {
            tensor.data[i + 1] = r * Math.sin(theta);
        }
    }
    
    return 0;
};

const tensorlib = {
    "new":       tensor_new,
    "get":       tensor_get,
    "set":       tensor_set,
    "matmul":    tensor_matmul,
    "add":       tensor_add,
    "mul":       tensor_mul,
    "sigmoid":   tensor_sigmoid,
    "relu":      tensor_relu,
    "bool":      tensor_bool,
    "shape":     tensor_shape,
    "transpose": tensor_transpose,
    "fill":      tensor_fill,
    "randn":     tensor_randn,
};

const luaopen_tensor = function(L) {
    lauxlib.luaL_newlib(L, tensorlib);
    
    // Create metatable for tensor
    lauxlib.luaL_newmetatable(L, to_luastring("tensor"));
    
    lua.lua_pushstring(L, to_luastring("__tostring"));
    lua.lua_pushcfunction(L, tensor_tostring);
    lua.lua_settable(L, -3);
    
    lua.lua_pushstring(L, to_luastring("__add"));
    lua.lua_pushcfunction(L, tensor_add);
    lua.lua_settable(L, -3);
    
    lua.lua_pushstring(L, to_luastring("__mul"));
    lua.lua_pushcfunction(L, tensor_mul);
    lua.lua_settable(L, -3);
    
    lua.lua_pop(L, 1);
    
    return 1;
};

module.exports.luaopen_tensor = luaopen_tensor;
module.exports.Tensor = Tensor;

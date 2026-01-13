"use strict";

const lua = require('../src/lua.js');
const lauxlib = require("../src/lauxlib.js");
const lualib = require("../src/lualib.js");
const {to_luastring} = require("../src/fengaricore.js");

test('tensorlogic library is available', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    lua.lua_getglobal(L, to_luastring("tensorlogic"));
    expect(lua.lua_istable(L, -1)).toBe(true);
});

test('tensorlogic relation creation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    const luaCode = `
        local relations = {
            {1, 2, 1.0},
            {2, 3, 1.0}
        }
        
        local rel = tensorlogic.relation(relations, 3)
        return tensor.get(rel, 1, 2), tensor.get(rel, 2, 3)
    `;
    
    expect(lauxlib.luaL_loadstring(L, to_luastring(luaCode))).toBe(lua.LUA_OK);
    lua.lua_call(L, 0, 2);
    
    expect(lua.lua_tonumber(L, -2)).toBe(1.0);
    expect(lua.lua_tonumber(L, -1)).toBe(1.0);
});

test('embedding creation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    const luaCode = `
        local emb = tensorlogic.embed(5, 10)
        local shape = tensor.shape(emb)
        return shape[1], shape[2]
    `;
    
    expect(lauxlib.luaL_loadstring(L, to_luastring(luaCode))).toBe(lua.LUA_OK);
    lua.lua_call(L, 0, 2);
    
    expect(lua.lua_tointeger(L, -2)).toBe(5);
    expect(lua.lua_tointeger(L, -1)).toBe(10);
});

test('similarity computation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    const luaCode = `
        local v1 = tensor.new({3})
        local v2 = tensor.new({3})
        
        -- Same vectors  
        tensor.set(v1, 1, 1.0)
        tensor.set(v1, 2, 0.0)
        tensor.set(v1, 3, 0.0)
        
        tensor.set(v2, 1, 1.0)
        tensor.set(v2, 2, 0.0)
        tensor.set(v2, 3, 0.0)
        
        local sim = tensorlogic.similarity(v1, v2)
        return sim
    `;
    
    expect(lauxlib.luaL_loadstring(L, to_luastring(luaCode))).toBe(lua.LUA_OK);
    lua.lua_call(L, 0, 1);
    
    expect(lua.lua_tonumber(L, -1)).toBeCloseTo(1.0, 5);
});

"use strict";

const lua = require('../src/lua.js');
const lauxlib = require("../src/lauxlib.js");
const lualib = require("../src/lualib.js");
const {to_luastring} = require("../src/fengaricore.js");

test('tensor library is available', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    lua.lua_getglobal(L, to_luastring("tensor"));
    expect(lua.lua_istable(L, -1)).toBe(true);
});

test('create and use tensor', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    const luaCode = `
        local t = tensor.new({2, 2})
        tensor.fill(t, 5.0)
        return tensor.get(t, 1, 1)
    `;
    
    expect(lauxlib.luaL_loadstring(L, to_luastring(luaCode))).toBe(lua.LUA_OK);
    lua.lua_call(L, 0, 1);
    
    expect(lua.lua_tonumber(L, -1)).toBe(5.0);
});

test('tensor matrix multiplication', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    const luaCode = `
        local t1 = tensor.new({2, 2})
        local t2 = tensor.new({2, 2})
        
        tensor.set(t1, 1, 1, 1)
        tensor.set(t1, 1, 2, 2)
        tensor.set(t1, 2, 1, 3)
        tensor.set(t1, 2, 2, 4)
        
        tensor.set(t2, 1, 1, 2)
        tensor.set(t2, 1, 2, 0)
        tensor.set(t2, 2, 1, 1)
        tensor.set(t2, 2, 2, 3)
        
        local t3 = tensor.matmul(t1, t2)
        return tensor.get(t3, 1, 1), tensor.get(t3, 2, 2)
    `;
    
    expect(lauxlib.luaL_loadstring(L, to_luastring(luaCode))).toBe(lua.LUA_OK);
    lua.lua_call(L, 0, 2);
    
    // t1 * t2 = [[1*2+2*1, 1*0+2*3], [3*2+4*1, 3*0+4*3]] = [[4, 6], [10, 12]]
    expect(lua.lua_tonumber(L, -2)).toBe(4);
    expect(lua.lua_tonumber(L, -1)).toBe(12);
});

test('tensor addition with metatable', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    const luaCode = `
        local t1 = tensor.new({2})
        local t2 = tensor.new({2})
        tensor.fill(t1, 2.0)
        tensor.fill(t2, 3.0)
        local t3 = t1 + t2
        return tensor.get(t3, 1)
    `;
    
    expect(lauxlib.luaL_loadstring(L, to_luastring(luaCode))).toBe(lua.LUA_OK);
    lua.lua_call(L, 0, 1);
    
    expect(lua.lua_tonumber(L, -1)).toBe(5.0);
});

test('tensor sigmoid activation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    const luaCode = `
        local t = tensor.new({1})
        tensor.set(t, 1, 0)
        local t2 = tensor.sigmoid(t)
        return tensor.get(t2, 1)
    `;
    
    expect(lauxlib.luaL_loadstring(L, to_luastring(luaCode))).toBe(lua.LUA_OK);
    lua.lua_call(L, 0, 1);
    
    expect(lua.lua_tonumber(L, -1)).toBeCloseTo(0.5, 5);
});

test('tensor boolean conversion', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    const luaCode = `
        local t = tensor.new({4})
        tensor.set(t, 1, 0.2)
        tensor.set(t, 2, 0.5)
        tensor.set(t, 3, 0.7)
        tensor.set(t, 4, 0.9)
        local t2 = tensor.bool(t)
        return tensor.get(t2, 1), tensor.get(t2, 3), tensor.get(t2, 4)
    `;
    
    expect(lauxlib.luaL_loadstring(L, to_luastring(luaCode))).toBe(lua.LUA_OK);
    lua.lua_call(L, 0, 3);
    
    expect(lua.lua_tonumber(L, -3)).toBe(0);
    expect(lua.lua_tonumber(L, -2)).toBe(1);
    expect(lua.lua_tonumber(L, -1)).toBe(1);
});

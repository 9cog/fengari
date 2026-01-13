"use strict";

const test_utils  = require("./tests.js");
const lua         = require("../src/lua.js");
const lauxlib     = require("../src/lauxlib.js");
const lualib      = require("../src/lualib.js");
const { to_luastring } = require("../src/fengaricore.js");

const toByteCode = test_utils.toByteCode;

test('tensor library is available', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    lua.lua_getglobal(L, to_luastring("tensor"));
    expect(lua.lua_istable(L, -1)).toBe(true);
});

test('create new tensor', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({3, 4})
            return t ~= nil
        `)
    );
    
    expect(lua.lua_toboolean(L, -1)).toBe(true);
});

test('tensor shape', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({2, 3})
            local shape = tensor.shape(t)
            return shape[1], shape[2]
        `)
    );
    
    expect(lua.lua_tointeger(L, -2)).toBe(2);
    expect(lua.lua_tointeger(L, -1)).toBe(3);
});

test('tensor set and get', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({2, 2})
            tensor.set(t, 1, 1, 5.0)
            tensor.set(t, 2, 2, 7.0)
            local v1 = tensor.get(t, 1, 1)
            local v2 = tensor.get(t, 2, 2)
            return v1, v2
        `)
    );
    
    expect(lua.lua_tonumber(L, -2)).toBe(5.0);
    expect(lua.lua_tonumber(L, -1)).toBe(7.0);
});

test('tensor fill', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({2, 2})
            tensor.fill(t, 3.5)
            return tensor.get(t, 1, 1), tensor.get(t, 2, 2)
        `)
    );
    
    expect(lua.lua_tonumber(L, -2)).toBe(3.5);
    expect(lua.lua_tonumber(L, -1)).toBe(3.5);
});

test('tensor addition', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t1 = tensor.new({2, 2})
            local t2 = tensor.new({2, 2})
            tensor.fill(t1, 2.0)
            tensor.fill(t2, 3.0)
            local t3 = tensor.add(t1, t2)
            return tensor.get(t3, 1, 1)
        `)
    );
    
    expect(lua.lua_tonumber(L, -1)).toBe(5.0);
});

test('tensor element-wise multiplication', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t1 = tensor.new({2, 2})
            local t2 = tensor.new({2, 2})
            tensor.fill(t1, 2.0)
            tensor.fill(t2, 3.0)
            local t3 = tensor.mul(t1, t2)
            return tensor.get(t3, 1, 1)
        `)
    );
    
    expect(lua.lua_tonumber(L, -1)).toBe(6.0);
});

test('tensor matrix multiplication', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t1 = tensor.new({2, 3})
            local t2 = tensor.new({3, 2})
            
            -- Fill t1: [[1, 2, 3], [4, 5, 6]]
            tensor.set(t1, 1, 1, 1)
            tensor.set(t1, 1, 2, 2)
            tensor.set(t1, 1, 3, 3)
            tensor.set(t1, 2, 1, 4)
            tensor.set(t1, 2, 2, 5)
            tensor.set(t1, 2, 3, 6)
            
            -- Fill t2: [[1, 2], [3, 4], [5, 6]]
            tensor.set(t2, 1, 1, 1)
            tensor.set(t2, 1, 2, 2)
            tensor.set(t2, 2, 1, 3)
            tensor.set(t2, 2, 2, 4)
            tensor.set(t2, 3, 1, 5)
            tensor.set(t2, 3, 2, 6)
            
            local t3 = tensor.matmul(t1, t2)
            -- Result should be [[22, 28], [49, 64]]
            return tensor.get(t3, 1, 1), tensor.get(t3, 1, 2),
                   tensor.get(t3, 2, 1), tensor.get(t3, 2, 2)
        `)
    );
    
    expect(lua.lua_tonumber(L, -4)).toBe(22);
    expect(lua.lua_tonumber(L, -3)).toBe(28);
    expect(lua.lua_tonumber(L, -2)).toBe(49);
    expect(lua.lua_tonumber(L, -1)).toBe(64);
});

test('tensor transpose', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t1 = tensor.new({2, 3})
            tensor.set(t1, 1, 1, 1)
            tensor.set(t1, 1, 2, 2)
            tensor.set(t1, 1, 3, 3)
            tensor.set(t1, 2, 1, 4)
            tensor.set(t1, 2, 2, 5)
            tensor.set(t1, 2, 3, 6)
            
            local t2 = tensor.transpose(t1)
            local shape = tensor.shape(t2)
            return shape[1], shape[2], 
                   tensor.get(t2, 1, 1), tensor.get(t2, 2, 1)
        `)
    );
    
    expect(lua.lua_tointeger(L, -4)).toBe(3);
    expect(lua.lua_tointeger(L, -3)).toBe(2);
    expect(lua.lua_tonumber(L, -2)).toBe(1);
    expect(lua.lua_tonumber(L, -1)).toBe(2);
});

test('tensor sigmoid activation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({2})
            tensor.set(t, 1, 0)
            tensor.set(t, 2, 1000)
            local t2 = tensor.sigmoid(t)
            local v1 = tensor.get(t2, 1)
            local v2 = tensor.get(t2, 2)
            return v1, v2
        `)
    );
    
    expect(lua.lua_tonumber(L, -2)).toBeCloseTo(0.5, 5);
    expect(lua.lua_tonumber(L, -1)).toBeCloseTo(1.0, 5);
});

test('tensor relu activation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({3})
            tensor.set(t, 1, -5)
            tensor.set(t, 2, 0)
            tensor.set(t, 3, 5)
            local t2 = tensor.relu(t)
            return tensor.get(t2, 1), tensor.get(t2, 2), tensor.get(t2, 3)
        `)
    );
    
    expect(lua.lua_tonumber(L, -3)).toBe(0);
    expect(lua.lua_tonumber(L, -2)).toBe(0);
    expect(lua.lua_tonumber(L, -1)).toBe(5);
});

test('tensor boolean conversion', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({4})
            tensor.set(t, 1, 0.2)
            tensor.set(t, 2, 0.5)
            tensor.set(t, 3, 0.7)
            tensor.set(t, 4, 0.9)
            local t2 = tensor.bool(t)
            return tensor.get(t2, 1), tensor.get(t2, 2), 
                   tensor.get(t2, 3), tensor.get(t2, 4)
        `)
    );
    
    expect(lua.lua_tonumber(L, -4)).toBe(0);
    expect(lua.lua_tonumber(L, -3)).toBe(0);
    expect(lua.lua_tonumber(L, -2)).toBe(1);
    expect(lua.lua_tonumber(L, -1)).toBe(1);
});

test('tensor random initialization', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({10})
            tensor.randn(t)
            local v1 = tensor.get(t, 1)
            local v2 = tensor.get(t, 2)
            return v1, v2
        `)
    );
    
    const v1 = lua.lua_tonumber(L, -2);
    const v2 = lua.lua_tonumber(L, -1);
    
    // Check that values are different and in reasonable range
    expect(v1).not.toBe(0);
    expect(v2).not.toBe(0);
    expect(v1).not.toBe(v2);
});

test('tensor metatable __add', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t1 = tensor.new({2})
            local t2 = tensor.new({2})
            tensor.fill(t1, 2.0)
            tensor.fill(t2, 3.0)
            local t3 = t1 + t2
            return tensor.get(t3, 1)
        `)
    );
    
    expect(lua.lua_tonumber(L, -1)).toBe(5.0);
});

test('tensor metatable __mul', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t1 = tensor.new({2})
            local t2 = tensor.new({2})
            tensor.fill(t1, 2.0)
            tensor.fill(t2, 3.0)
            local t3 = t1 * t2
            return tensor.get(t3, 1)
        `)
    );
    
    expect(lua.lua_tonumber(L, -1)).toBe(6.0);
});

test('tensor tostring', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({2, 3})
            tensor.fill(t, 1.5)
            return tostring(t)
        `)
    );
    
    const str = lua.lua_tojsstring(L, -1);
    expect(str).toContain("Tensor");
    expect(str).toContain("shape=[2, 3]");
});

test('tensor error on mismatched shapes', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    expect(() => {
        test_utils.lauxlib_dostring(
            L,
            to_luastring(`
                local t1 = tensor.new({2, 2})
                local t2 = tensor.new({3, 3})
                local t3 = tensor.add(t1, t2)
            `)
        );
    }).toThrow();
});

test('tensor error on invalid matmul dimensions', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    expect(() => {
        test_utils.lauxlib_dostring(
            L,
            to_luastring(`
                local t1 = tensor.new({2, 3})
                local t2 = tensor.new({2, 2})
                local t3 = tensor.matmul(t1, t2)
            `)
        );
    }).toThrow();
});

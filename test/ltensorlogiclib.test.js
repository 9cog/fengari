"use strict";

const test_utils  = require("./tests.js");
const lua         = require("../src/lua.js");
const lauxlib     = require("../src/lauxlib.js");
const lualib      = require("../src/lualib.js");
const { to_luastring } = require("../src/fengaricore.js");

test('tensorlogic library is available', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    lua.lua_getglobal(L, to_luastring("tensorlogic"));
    expect(lua.lua_istable(L, -1)).toBe(true);
});

test('create knowledge base', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local kb = tensorlogic.kb_new()
            return kb ~= nil
        `)
    );
    
    expect(lua.lua_toboolean(L, -1)).toBe(true);
});

test('knowledge base add and get predicate', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local kb = tensorlogic.kb_new()
            local t = tensor.new({2, 2})
            tensor.fill(t, 1.0)
            
            tensorlogic.kb_add(kb, "parent", t)
            local retrieved = tensorlogic.kb_get(kb, "parent")
            
            return retrieved ~= nil, tensor.get(retrieved, 1, 1)
        `)
    );
    
    expect(lua.lua_toboolean(L, -2)).toBe(true);
    expect(lua.lua_tonumber(L, -1)).toBe(1.0);
});

test('knowledge base mode setting', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local kb = tensorlogic.kb_new()
            local mode1 = tensorlogic.kb_get_mode(kb)
            
            tensorlogic.kb_mode(kb, "continuous")
            local mode2 = tensorlogic.kb_get_mode(kb)
            
            return mode1, mode2
        `)
    );
    
    expect(lua.lua_tojsstring(L, -2)).toBe("boolean");
    expect(lua.lua_tojsstring(L, -1)).toBe("continuous");
});

test('logical AND operation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t1 = tensor.new({2})
            local t2 = tensor.new({2})
            tensor.set(t1, 1, 1.0)
            tensor.set(t1, 2, 0.0)
            tensor.set(t2, 1, 1.0)
            tensor.set(t2, 2, 1.0)
            
            local result = tensorlogic.land(t1, t2)
            return tensor.get(result, 1), tensor.get(result, 2)
        `)
    );
    
    expect(lua.lua_tonumber(L, -2)).toBeGreaterThan(0.5);
    expect(lua.lua_tonumber(L, -1)).toBe(0);
});

test('logical OR operation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t1 = tensor.new({2})
            local t2 = tensor.new({2})
            tensor.set(t1, 1, 1.0)
            tensor.set(t1, 2, 0.0)
            tensor.set(t2, 1, 0.0)
            tensor.set(t2, 2, 0.0)
            
            local result = tensorlogic.lor(t1, t2)
            return tensor.get(result, 1), tensor.get(result, 2)
        `)
    );
    
    expect(lua.lua_tonumber(L, -2)).toBeGreaterThan(0.5);
    expect(lua.lua_tonumber(L, -1)).toBe(0);
});

test('logical NOT operation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({2})
            tensor.set(t, 1, 1.0)
            tensor.set(t, 2, 0.0)
            
            local result = tensorlogic.lnot(t)
            return tensor.get(result, 1), tensor.get(result, 2)
        `)
    );
    
    expect(lua.lua_tonumber(L, -2)).toBe(0);
    expect(lua.lua_tonumber(L, -1)).toBe(1);
});

test('relational composition (grandparent rule)', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            -- Create parent relation: 3 people
            -- 0 -> 1 (0 is parent of 1)
            -- 1 -> 2 (1 is parent of 2)
            local parent = tensor.new({3, 3})
            tensor.set(parent, 1, 2, 1.0)  -- 0 is parent of 1
            tensor.set(parent, 2, 3, 1.0)  -- 1 is parent of 2
            
            -- Compose to get grandparent
            local grandparent = tensorlogic.compose(parent, parent)
            
            -- 0 should be grandparent of 2
            return tensor.get(grandparent, 1, 3)
        `)
    );
    
    expect(lua.lua_tonumber(L, -1)).toBeGreaterThan(0.5);
});

test('create relation from table', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local relations = {
                {1, 2, 1.0},
                {2, 3, 1.0},
                {3, 1, 0.5}
            }
            
            local rel = tensorlogic.relation(relations, 3)
            return tensor.get(rel, 1, 2), tensor.get(rel, 2, 3), tensor.get(rel, 3, 1)
        `)
    );
    
    expect(lua.lua_tonumber(L, -3)).toBe(1.0);
    expect(lua.lua_tonumber(L, -2)).toBe(1.0);
    expect(lua.lua_tonumber(L, -1)).toBe(0.5);
});

test('query predicate with threshold', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t = tensor.new({4})
            tensor.set(t, 1, 0.2)
            tensor.set(t, 2, 0.6)
            tensor.set(t, 3, 0.8)
            tensor.set(t, 4, 0.3)
            
            local results = tensorlogic.query(t, 0.5)
            local count = 0
            for k, v in pairs(results) do
                count = count + 1
            end
            return count, results[2], results[3]
        `)
    );
    
    expect(lua.lua_tointeger(L, -3)).toBe(2);
    expect(lua.lua_tonumber(L, -2)).toBe(0.6);
    expect(lua.lua_tonumber(L, -1)).toBe(0.8);
});

test('create embeddings', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local emb = tensorlogic.embed(5, 10)
            local shape = tensor.shape(emb)
            return shape[1], shape[2]
        `)
    );
    
    expect(lua.lua_tointeger(L, -2)).toBe(5);
    expect(lua.lua_tointeger(L, -1)).toBe(10);
});

test('similarity computation', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local v1 = tensor.new({3})
            local v2 = tensor.new({3})
            local v3 = tensor.new({3})
            
            -- Same vectors
            tensor.set(v1, 1, 1.0)
            tensor.set(v1, 2, 0.0)
            tensor.set(v1, 3, 0.0)
            
            tensor.set(v2, 1, 1.0)
            tensor.set(v2, 2, 0.0)
            tensor.set(v2, 3, 0.0)
            
            -- Orthogonal vector
            tensor.set(v3, 1, 0.0)
            tensor.set(v3, 2, 1.0)
            tensor.set(v3, 3, 0.0)
            
            local sim1 = tensorlogic.similarity(v1, v2)
            local sim2 = tensorlogic.similarity(v1, v3)
            
            return sim1, sim2
        `)
    );
    
    expect(lua.lua_tonumber(L, -2)).toBeCloseTo(1.0, 5);
    expect(lua.lua_tonumber(L, -1)).toBeCloseTo(0.0, 5);
});

test('complex logical reasoning', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            -- Test: (A AND B) OR (NOT C)
            local a = tensor.new({3})
            local b = tensor.new({3})
            local c = tensor.new({3})
            
            tensor.set(a, 1, 1.0)
            tensor.set(a, 2, 0.0)
            tensor.set(a, 3, 1.0)
            
            tensor.set(b, 1, 1.0)
            tensor.set(b, 2, 1.0)
            tensor.set(b, 3, 0.0)
            
            tensor.set(c, 1, 0.0)
            tensor.set(c, 2, 0.0)
            tensor.set(c, 3, 1.0)
            
            local ab = tensorlogic.land(a, b)
            local not_c = tensorlogic.lnot(c)
            local result = tensorlogic.lor(ab, not_c)
            
            return tensor.get(result, 1), tensor.get(result, 2), tensor.get(result, 3)
        `)
    );
    
    // Result[1] should be high (1 AND 1 OR NOT 0 = 1 OR 1)
    // Result[2] should be high (0 AND 1 OR NOT 0 = 0 OR 1)
    // Result[3] should be moderate (1 AND 0 OR NOT 1 = 0 OR 0)
    expect(lua.lua_tonumber(L, -3)).toBeGreaterThan(0.5);
    expect(lua.lua_tonumber(L, -2)).toBeGreaterThan(0.5);
    expect(lua.lua_tonumber(L, -1)).toBeLessThan(0.5);
});

test('multi-hop reasoning with composition', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            -- Create a simple family tree
            -- Person 0 -> Person 1 -> Person 2 -> Person 3
            local parent = tensor.new({4, 4})
            tensor.set(parent, 1, 2, 1.0)  -- 0 parent of 1
            tensor.set(parent, 2, 3, 1.0)  -- 1 parent of 2
            tensor.set(parent, 3, 4, 1.0)  -- 2 parent of 3
            
            -- Grandparent
            local grandparent = tensorlogic.compose(parent, parent)
            
            -- Great-grandparent
            local greatgrandparent = tensorlogic.compose(grandparent, parent)
            
            -- Person 0 should be great-grandparent of Person 3
            return tensor.get(greatgrandparent, 1, 4)
        `)
    );
    
    expect(lua.lua_tonumber(L, -1)).toBeGreaterThan(0.5);
});

test('boolean vs continuous mode semantics', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            local t1 = tensor.new({2})
            local t2 = tensor.new({2})
            
            tensor.set(t1, 1, 0.7)
            tensor.set(t1, 2, 0.3)
            tensor.set(t2, 1, 0.8)
            tensor.set(t2, 2, 0.9)
            
            -- AND operation should give intermediate values in continuous mode
            local result = tensorlogic.land(t1, t2)
            
            -- Convert to boolean mode
            local bool_result = tensor.bool(result, 0.5)
            
            return tensor.get(result, 1), tensor.get(result, 2),
                   tensor.get(bool_result, 1), tensor.get(bool_result, 2)
        `)
    );
    
    const cont1 = lua.lua_tonumber(L, -4);
    const cont2 = lua.lua_tonumber(L, -3);
    const bool1 = lua.lua_tonumber(L, -2);
    const bool2 = lua.lua_tonumber(L, -1);
    
    // In continuous mode, values should be intermediate
    expect(cont1).toBeGreaterThan(0.4);
    expect(cont1).toBeLessThan(1.0);
    
    // In boolean mode, values should be 0 or 1
    expect(bool1).toBe(1);
    expect(bool2).toBe(0);
});

test('probabilistic reasoning with soft values', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    test_utils.lauxlib_dostring(
        L,
        to_luastring(`
            -- Uncertain knowledge: person might be parent
            local uncertain_parent = tensor.new({2, 2})
            tensor.set(uncertain_parent, 1, 2, 0.7)  -- 70% confidence
            
            -- Compose uncertain relations
            local uncertain_grandparent = tensorlogic.compose(uncertain_parent, uncertain_parent)
            
            -- Result should reflect uncertainty propagation
            return tensor.get(uncertain_grandparent, 1, 2)
        `)
    );
    
    const result = lua.lua_tonumber(L, -1);
    // With composition of 0.7, we expect some uncertainty propagation
    expect(result).toBeGreaterThan(0);
    expect(result).toBeLessThan(1.0);
});

test('error on invalid mode', () => {
    let L = lauxlib.luaL_newstate();
    lualib.luaL_openlibs(L);
    
    expect(() => {
        test_utils.lauxlib_dostring(
            L,
            to_luastring(`
                local kb = tensorlogic.kb_new()
                tensorlogic.kb_mode(kb, "invalid_mode")
            `)
        );
    }).toThrow();
});

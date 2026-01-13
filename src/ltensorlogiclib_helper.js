// Helper functions for tensor logic library

const lua     = require('./lua.js');
const lauxlib = require('./lauxlib.js');
const { to_luastring } = require('./fengaricore.js');

// Get tensor from userdata
const gettensor = function(L, idx) {
    const udata = lua.lua_touserdata(L, idx);
    if (!udata || !udata.tensor) {
        lauxlib.luaL_error(L, to_luastring("expected tensor"));
        return null;
    }
    return udata.tensor;
};

// Get knowledge base from userdata
const getkb = function(L, idx) {
    const udata = lua.lua_touserdata(L, idx);
    if (!udata || !udata.kb) {
        lauxlib.luaL_error(L, to_luastring("expected knowledge base"));
        return null;
    }
    return udata.kb;
};

module.exports.gettensor = gettensor;
module.exports.getkb = getkb;

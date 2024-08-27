local stringify = require("pandoc.utils").stringify

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
 end

if FORMAT:match 'revealjs' then
    function Image(s, src)

        local iname = stringify(s.src)
        local _, _, filename, ext = iname:find("^(.+)(%..+)$")
        if ((ext == ".png" or ext == ".jpeg" or ext == ".jpg") and not (s.attr.classes[1] == "notransparent")) then
            local fname = string.format('%s-transparent.png', filename)
            if not file_exists(fname) then
                local cmd = io.popen(string.format('magick convert %s -fuzz %s -transparent white %s', iname, '2%', fname))
                if (cmd) then
                    cmd:read("a")
                    cmd:close()                    
                end
            end
            return pandoc.Image(s.caption, fname, s.title, s.attr)
        end
    end
end

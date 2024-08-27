local prefix = "docs-resources/Slides"

function Meta(meta)
  if quarto.format.is_revealjs_output() then
    quarto.log.debug("Not Skipping meta filter for solution file")
    local filename = pandoc.path.join({quarto.project.directory or ".", 'slides.yml'})
    local file = io.open(filename, 'a')
    if file then
      local title = pandoc.utils.stringify(meta.title)
      -- get the ouput file name
      local html = quarto.doc.project_output_file()
      local dir, filename = string.match(quarto.doc.input_file, "(.-)([^\\/]-%.?([^%.\\/]*))$")
      if html ~= nil then
        -- replace all backslashes with forward slashes
        html = html:gsub("\\", "/")
        file:write("- title: \"" .. title .. "\"\n")
        file:write("  html: \"" .. prefix .. "/" .. html .. "\"\n")
        file:write("  name: \"" .. filename .. "\"\n")
      end
      file:close()
    end
  end
end
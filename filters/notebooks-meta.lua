local prefix = "docs-resources/Notebooks"

function Meta(meta)
  local dir = string.match(quarto.doc.input_file, "(.-)([^\\/]-%.?([^%.\\/]*))$")
  local dir = dir:gsub("\\", "/")
  if dir:match("Solutions/$") and quarto.format.is_html_output() then
    local filename = pandoc.path.join({quarto.project.directory or ".", 'applications.yml'})
    local file = io.open(filename, 'a')
    if file then
      local title = pandoc.utils.stringify(meta.title)
      local orig = quarto.doc.project_output_file()
      if orig ~= nil then
        orig = orig:gsub("\\", "/")
        appfile = orig:gsub("Solutions/", "Applications/")
        local html = orig
        local py = appfile:gsub(".html", ".py")
        local ipynb = html:gsub(".html", ".ipynb")
        file:write("- title: \"" .. title .. "\"\n")
        if quarto.doc.input_file:match("%-sol%.qmd$") then
          py = py:gsub("-sol.py", ".py")
          file:write("  html: \"" .. prefix .. "/" .. html .. "\"\n")
          file:write("  ipynb: \"" .. prefix .. "/" .. ipynb .. "\"\n")
        end
        file:write("  py: \"" .. prefix .. "/" .. py .. "\"\n")
      end
    end
  end
end
function Div(element)
  -- only execute this filter on files not ending with "-sol.qmd"
  if quarto.doc.input_file:match("-sol%.qmd$") ~= nil then
    return nil
  end
  if element.classes[1] == "cell" then
    if element.attr.attributes["tags"] ~= nil then
      tags = element.attr.attributes["tags"]
      -- find if there is the "solution" tag
      if tags:find("\"solution\"") then
        quarto.log.debug("Found cell solution")
        return {}
       end
      end
  elseif element.classes[1] == "solution" then
    quarto.log.debug("Found solution")
    return {}
  end
end
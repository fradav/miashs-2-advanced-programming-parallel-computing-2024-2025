function Header(element)

  -- if element.level == 2 and re.match(pandoc.utils.stringify(element),"\\(cont.\\)?\\s+(?\\d)?\\s*$")  then
  if element.level == 2 then
    txt = pandoc.utils.stringify(element)
    -- match a header that ends with "cont. 1" or "cont." or "(1)"
    if txt:match("cont%.%s+%d%s*$") or txt:match("cont%.%s*$") or txt:match("%(%d%)%s*$") then
      quarto.log.debug("Removing slide header: " .. txt)
      return {}
    end
  end
end

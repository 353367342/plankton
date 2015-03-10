function setAug(a)
	local noaug = false
	local fName = {}
	if a == 1 then
		fName = 'noaug1'
	elseif a == 2 then
		fName = 'noaug2'
	end
	if file_exists(fName) then
		local f = io.open(fName,'r')
		local starg = f:read('*line')
		f:close()
		os.remove(fName)
		if starg == 'true' then
			noaug = true
		else 
			noaug = false
		end
	end
	return noaug
end

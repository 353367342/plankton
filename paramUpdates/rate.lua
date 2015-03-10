function setRate(a)
	local rate = optimState.learningRate
	local fName = {}
	if a == 1 then
		fName = 'rate1'
	elseif a == 2 then
		fName = 'rate2'
	end
	if file_exists(fName) then
		local f = io.open(fName,'r')
		rate = tonumber(f:read('*line'))
		f:close()
		os.remove(fName)
	end
	return rate
end

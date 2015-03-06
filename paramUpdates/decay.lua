function setDecay(a)
	local decay = optimState.weightDecay
	local fName = {}
	if a == 1 then
		fName = 'rate1'
	elseif a == 2 then
		fName = 'rate2'
	end
	if file_exists(fName) then
		local f = io.open(fName,'r')
		decay = tonumber(f:read('*line'))
		f:close()
		os.remove(fName)
	end
	return decay
end

function setRate()
	local rate = optimState.learningRate
	if file_exists('rate') then
		local f = io.open('rate','r')
		rate = tonumber(f:read('*line'))
		f:close()
		os.remove('rate')
	end
	return rate
end
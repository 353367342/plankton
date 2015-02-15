function setDecay()
	local decay = optimState.weightDecay
	if file_exists('decay') then
		local f = io.open('decay','r')
		decay = tonumber(f:read('*line'))
		f:close()
		os.remove('decay')
	end
	return decay
end

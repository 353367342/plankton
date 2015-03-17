function setAug()
	local noaug = false
	local fName = 'noaug'
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

function setDecay()
	local decay = optimState.weightDecay
	local fName = 'decay'
	if file_exists(fName) then
		local f = io.open(fName,'r')
		decay = tonumber(f:read('*line'))
		f:close()
		os.remove(fName)
	end
	return decay
end

function setRate()
	local rate = optimState.learningRate
	local fName = 'rate'
	if file_exists(fName) then
		local f = io.open(fName,'r')
		rate = tonumber(f:read('*line'))
		f:close()
		os.remove(fName)
	end
	return rate
end

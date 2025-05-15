#!/usr/bin/env luajit

local template = require 'template'
local ffi = require 'ffi'
local matrix = require 'matrix'
local gnuplot = require 'gnuplot'
require 'ext'


local function cexp(theta)
	return matrix{math.cos(theta), math.sin(theta)}
end


local n = 10
--local disks = range(n):map(function(i) return {dist=1, radius=1e-3, angle=math.pi*((i+.5)/n + .5)} end)
--local disks = range(n):map(function(i) return {dist=1-.5*i/n, radius=1e-3, angle=math.pi * (.5 + (i-.5)/n)} end)
local disks = range(n):map(function(i) return {dist=1-.5*i/n, radius=.1*2^(-i), angle=2 * math.pi * (i-.5)/n} end)

local size = matrix{1024,1024}
local xmin = matrix{-2.5, -2.5}
local xmax = matrix{2.5, 2.5}
local pointChargeDensity = 2 * math.pi




print'creating env...'
local env = require 'cl.obj.env'{size=size, precision='float'}
local dx = (xmax - xmin):ediv(size)
local dxlen = dx:norm()


local function fromBinFile(name)
	local dstPtr = ffi.new(env.real..'[?]', env.base.volume)
	local filename = name..'.'..env.real
	local srcStr = assert(path(filename):read(), "failed to load file "..filename)
	local srcPtr = ffi.cast('char*', srcStr)
	ffi.copy(dstPtr, srcPtr, ffi.sizeof'real' * env.base.volume)
	return dstPtr
end


-- [===[	enable this to regenerate the data


local rhoCPU = ffi.new('real[?]', env.base.volume)	-- this will be a point charge
local phiCPU = ffi.new('real[?]', env.base.volume)	-- the is the potential
local boundaryCPU = ffi.new('real[?]', env.base.volume)	-- the is the boundary flag - multiplied per-iteration.  1 where things are good, 0 at boundary.



for i=0,env.base.volume-1 do
	rhoCPU[i] = 0
end

print'writing boundary...'
do
	-- 	
	for i=0,env.base.volume-1 do
		boundaryCPU[i] = 1
	end

	-- point charge
	for i=0,env.base.volume-1 do
		rhoCPU[i] = 0
	end
	local p = (matrix{2,0} - xmin):ediv(xmax - xmin):emul(size-1):map(math.floor)
	assert(p[1] >= 0 and p[1] < size[1] and p[2] >= 0 and p[2] < size[2], "looks like your point charge source is out of the domain")
	rhoCPU[ p[1] + size[1] * p[2] ] = pointChargeDensity / dx:prod()

	local oob
	for j,disk in ipairs(disks) do
		print(tolua(disk))
		local center = cexp(disk.angle) * disk.dist

		local rdivs = math.ceil(disk.radius * 2 / dxlen) + 1
		for ir=0,rdivs-1 do
			local r = (ir+.5)/rdivs * disk.radius 
			local thdivs = math.ceil(2 * math.pi * r * 2/dxlen) + 1		-- radial divisions, stepping by half a pixel
			for ith=0,thdivs-1 do	-- TODO circumference of radius
				local th = 2 * math.pi * (ith+.5) / thdivs
				local p = (cexp(th) * r + center - xmin):ediv(xmax - xmin):emul(size-1):map(math.floor)
				if p[1] >= 0 and p[1] < size[1] and p[2] >= 0 and p[2] < size[2] then
					local index = p[1] + size[1] * p[2]
					boundaryCPU[index] = 0
				else
					oob = true
				end
			end
		end
	end
	if oob then
		io.stderr:write'DANGER! boundary conditions extend off of domain\n'
	end
end

--if path('phi.'..env.real):exists() then
--	phiCPU = fromBinFile'phi'
--else
	local boundaryPotential = 1
	for i=0,env.base.volume-1 do
		phiCPU[i] = boundaryCPU[i] * -rhoCPU[i]
			+ (1 - boundaryCPU[i]) * boundaryPotential 
	end
--end

local rho = env:buffer{name='rho', data=rhoCPU}
local phi = env:buffer{name='phi', data=phiCPU}
local boundary = env:buffer{name='boundary', data=boundaryCPU}

local A = env:kernel{
	argsOut = {rho},
	argsIn = {phi, boundary},
	body = template([[	
#if 0	//set boundary to zero...  this only must be done if there are no inter-domain boundary constraints (where boundary[index] == 0)
	if (i.x == 0 || i.x >= size.x-1 ||
		i.y == 0 || i.y >= size.y-1)
	{
		rho[index] = 0;	//boundary conditions
		return;
	} 
#endif
	
	real sum = 0.;
	
	<? for k=0,dim-1 do ?>{
		int4 iL = i;
		iL.s<?=k?> = max(0, iL.s<?=k?>-1);
		int indexL = indexForInt4(iL);

		int4 iR = i;
		iR.s<?=k?> = min(<?=(size[k+1]-1)?>, iR.s<?=k?>+1);
		int indexR = indexForInt4(iR);

		sum += (phi[indexR] - 2. * phi[index] + phi[indexL]) * <?=1 / (dx[k+1] * dx[k+1]) ?>;
	}<? end ?>
	
	rho[index] = sum * boundary[index];
	
	//rho[index] += (1. - boundary[index]) * phi[index];	//this should fix phi to zero at boundary=0, right? or do the opposite?
]], {
	dx = dx,
	dim = env.base.dim,
	size = size,
	clnumber = require 'cl.obj.number',
})}

local solver = 
--require 'solver.cl.conjgrad'
require 'solver.cl.conjres'
--require 'solver.cl.gmres'
{
	env = env,
	A = A,
	b = rho,
	x = phi,
	epsilon = 1e-7,
	errorCallback = function(err,iter)
		io.stderr:write(tostring(err)..'\t'..tostring(iter)..'\n')
		assert(err == err)
	end,
}

local beginTime = os.clock()
solver()
local endTime = os.clock()
print('took '..(endTime - beginTime)..' seconds')

rho:toCPU(rhoCPU)
phi:toCPU(phiCPU)

print'writing results...'

local function toBinFile(name, data)
	path(name..'.'..env.real):write(ffi.string(ffi.cast('char*', data), ffi.sizeof'real' * env.base.volume))
end
toBinFile('rho', rhoCPU)
toBinFile('phi', phiCPU)
toBinFile('boundary', boundaryCPU)

--]===]
-- [===[


local rhoCPU = fromBinFile'rho'
local phiCPU = fromBinFile'phi'
local boundaryCPU = fromBinFile'boundary'

do
	local i = (matrix{0,0} - xmin):ediv(xmax - xmin):emul(size-1):map(math.floor)
	local phiR = matrix{
		phiCPU[i[1]+1 + size[1] * i[2]],
		phiCPU[i[1] + size[1] * (i[2]+1)]
	}
	local phiL = matrix{
		phiCPU[i[1]-1 + size[1] * i[2]],
		phiCPU[i[1] + size[1] * (i[2]-1)]
	}
	local dphi = (phiR - phiL):ediv(2 * dx)
	local magn_grad_phi = dphi:norm() 
	print('|grad phi(0)|', magn_grad_phi)
end

--]===]


local cbmin, cbmax = 0, 1.1

-- [[ draw a picture
local colors = {	
	{0x00,0x00,0xff},
	{0x00,0x59,0xfb},
	{0x00,0xb5,0xeb},
	{0x22,0xec,0xd3},
	{0x82,0xff,0xb3},
	{0xe2,0xea,0x8c},
	{0xff,0xb2,0x60},
	{0xff,0x67,0x35},
	{0xff,0x00,0x00},
}
local function gradient(value)
	value = value % 1 
	value = value * (1 - 2e-7) + 1e-7
	value = value * (#colors - 1)
	local i1 = math.floor(value) + 1
	local i2 = i1 % #colors + 1
	local f = (value + 1) - i1
	return (matrix(colors[i1]) * (1 - f) 
		+ matrix(colors[i2]) * f):unpack()
end
function drawRaw(name, buffer)
	require 'image'(size[1], size[2], 3, 'unsigned char', function(i,j)
		local value = (math.abs(buffer[i + size[1] * j]) - cbmin) / (cbmax - cbmin)
		--value = math.clamp(value, 0, 1)
		return gradient(value)
	end):save(name..'-out-raw.png')
end
drawRaw('phi', phiCPU)
drawRaw('boundary', boundaryCPU)
--]]



--[[ enable this to regenerate out.txt
local file = assert(path'out.txt':open'wb')
file:write('#x y rho phi boundary\n')
local index = 0
for j=0,size[2]-1 do
	for i=0,size[1]-1 do
		file:write(
			(i+.5)/size[1]*(xmax[1] - xmin[1]) + xmin[1],
			'\t',(j+.5)/size[2]*(xmax[2] - xmin[2]) + xmin[2],
			'\t',rhoCPU[index],
			'\t',phiCPU[index],
			'\t',boundaryCPU[index],
			'\n')
		index = index + 1
	end
	file:write'\n'
end
file:close()
print'done!'
--]]

--[[
gnuplot{
	persist = true,
	style = 'data lines',
	{splot=true, datafile='out.txt', using='1:2:3', title='rho'},
}
--]]
--[[
gnuplot{
	persist = true,
	style = 'data lines',
	view = 'map',
	unset = {
		'key', 
		'surface',
	},
	contour = true,
	cntrparam = 'levels 10000',
	--log = 'z',
	xrange = {-1,0},
	{splot=true, datafile='out.txt', using='1:2:(abs($4))', title='phi'},
}
--]]
--[[
gnuplot{
	terminal = 'png size 1024,768',
	output='out-gnuplot.png', --persist = true,
	size = 'square',
	pm3d = 'map',
	palette = 'rgbformulae 22,13,10',
	cbrange = {cbmin,cbmax},
	{splot=true, datafile='out.txt', using='1:2:(abs($4))', title='phi'},
}
--]]
--[[
gnuplot{
	persist = true,
	style = 'data lines',
	{splot=true, datafile='out.txt', using='1:2:5', title='boundary'},
}
--]]

path'out.txt':remove()	-- 50mbs 

'use strict'

const M = Math,
	D = document,
	W = window,
	FA = Float32Array,
	idMat = new FA([
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1]),
	projMat = new FA(idMat),
	viewMat = new FA(idMat),
	modelViewMat = new FA(16),
	findMat = new FA(16),
	horizon = 50,
	lightProjMat = new FA(idMat),
	lightViewMat = new FA(idMat),
	skyColor = [.06, .06, .06, 1],
	offscreenWidth = 256,
	offscreenHeight = 256,
	shadowDepthTextureSize = 1024

let gl,
	shadowFramebuffer,
	shadowDepthTexture,
	shadowProgram,
	offscreenBuffer,
	offscreenTexture,
	offscreenProgram,
	screenVertexBuffer,
	screenTextureBuffer,
	screenProgram,
	entitiesLength,
	entities = [],
	width,
	height,
	ymax,
	widthToGl,
	heightToGl,
	pointersLength,
	pointersX = [],
	pointersY = [],
	keysDown = []

M.PI2 = M.PI2 || M.PI / 2
M.TAU = M.TAU || M.PI * 2

// from https://github.com/toji/gl-matrix
function invert(out, a) {
	const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3],
		a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7],
		a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11],
		a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15],
		b00 = a00 * a11 - a01 * a10,
		b01 = a00 * a12 - a02 * a10,
		b02 = a00 * a13 - a03 * a10,
		b03 = a01 * a12 - a02 * a11,
		b04 = a01 * a13 - a03 * a11,
		b05 = a02 * a13 - a03 * a12,
		b06 = a20 * a31 - a21 * a30,
		b07 = a20 * a32 - a22 * a30,
		b08 = a20 * a33 - a23 * a30,
		b09 = a21 * a32 - a22 * a31,
		b10 = a21 * a33 - a23 * a31,
		b11 = a22 * a33 - a23 * a32

	// calculate the determinant
	let d = b00 * b11 -
			b01 * b10 +
			b02 * b09 +
			b03 * b08 -
			b04 * b07 +
			b05 * b06

	if (!d) {
		return
	}

	d = 1.0 / d

	out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * d
	out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * d
	out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * d
	out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * d
	out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * d
	out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * d
	out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * d
	out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * d
	out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * d
	out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * d
	out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * d
	out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * d
	out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * d
	out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * d
	out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * d
	out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * d
}

// from https://github.com/toji/gl-matrix
function multiply(out, a, b) {
	let a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3],
		a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7],
		a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11],
		a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15]

	// cache only the current line of the second matrix
	let b0  = b[0], b1 = b[1], b2 = b[2], b3 = b[3]
	out[0] = b0*a00 + b1*a10 + b2*a20 + b3*a30
	out[1] = b0*a01 + b1*a11 + b2*a21 + b3*a31
	out[2] = b0*a02 + b1*a12 + b2*a22 + b3*a32
	out[3] = b0*a03 + b1*a13 + b2*a23 + b3*a33

	b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7]
	out[4] = b0*a00 + b1*a10 + b2*a20 + b3*a30
	out[5] = b0*a01 + b1*a11 + b2*a21 + b3*a31
	out[6] = b0*a02 + b1*a12 + b2*a22 + b3*a32
	out[7] = b0*a03 + b1*a13 + b2*a23 + b3*a33

	b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11]
	out[8] = b0*a00 + b1*a10 + b2*a20 + b3*a30
	out[9] = b0*a01 + b1*a11 + b2*a21 + b3*a31
	out[10] = b0*a02 + b1*a12 + b2*a22 + b3*a32
	out[11] = b0*a03 + b1*a13 + b2*a23 + b3*a33

	b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15]
	out[12] = b0*a00 + b1*a10 + b2*a20 + b3*a30
	out[13] = b0*a01 + b1*a11 + b2*a21 + b3*a31
	out[14] = b0*a02 + b1*a12 + b2*a22 + b3*a32
	out[15] = b0*a03 + b1*a13 + b2*a23 + b3*a33
}

// from https://github.com/toji/gl-matrix
function rotate(out, a, rad, x, y, z) {
	let len = M.sqrt(x * x + y * y + z * z),
		s, c, t,
		a00, a01, a02, a03,
		a10, a11, a12, a13,
		a20, a21, a22, a23,
		b00, b01, b02,
		b10, b11, b12,
		b20, b21, b22

	if (M.abs(len) < 0.000001) {
		return
	}

	len = 1 / len
	x *= len
	y *= len
	z *= len

	s = M.sin(rad)
	c = M.cos(rad)
	t = 1 - c

	a00 = a[0]; a01 = a[1]; a02 = a[2]; a03 = a[3]
	a10 = a[4]; a11 = a[5]; a12 = a[6]; a13 = a[7]
	a20 = a[8]; a21 = a[9]; a22 = a[10]; a23 = a[11]

	// construct the elements of the rotation matrix
	b00 = x * x * t + c; b01 = y * x * t + z * s; b02 = z * x * t - y * s
	b10 = x * y * t - z * s; b11 = y * y * t + c; b12 = z * y * t + x * s
	b20 = x * z * t + y * s; b21 = y * z * t - x * s; b22 = z * z * t + c

	// perform rotation-specific matrix multiplication
	out[0] = a00 * b00 + a10 * b01 + a20 * b02
	out[1] = a01 * b00 + a11 * b01 + a21 * b02
	out[2] = a02 * b00 + a12 * b01 + a22 * b02
	out[3] = a03 * b00 + a13 * b01 + a23 * b02
	out[4] = a00 * b10 + a10 * b11 + a20 * b12
	out[5] = a01 * b10 + a11 * b11 + a21 * b12
	out[6] = a02 * b10 + a12 * b11 + a22 * b12
	out[7] = a03 * b10 + a13 * b11 + a23 * b12
	out[8] = a00 * b20 + a10 * b21 + a20 * b22
	out[9] = a01 * b20 + a11 * b21 + a21 * b22
	out[10] = a02 * b20 + a12 * b21 + a22 * b22
	out[11] = a03 * b20 + a13 * b21 + a23 * b22

	if (a !== out) {
		// if the source and destination differ, copy the unchanged last row
		out[12] = a[12]
		out[13] = a[13]
		out[14] = a[14]
		out[15] = a[15]
	}
}

// from https://github.com/toji/gl-matrix
function scale(out, a, x, y, z) {
	out[0] = a[0] * x
	out[1] = a[1] * x
	out[2] = a[2] * x
	out[3] = a[3] * x
	out[4] = a[4] * y
	out[5] = a[5] * y
	out[6] = a[6] * y
	out[7] = a[7] * y
	out[8] = a[8] * z
	out[9] = a[9] * z
	out[10] = a[10] * z
	out[11] = a[11] * z
	out[12] = a[12]
	out[13] = a[13]
	out[14] = a[14]
	out[15] = a[15]
}

// from https://github.com/toji/gl-matrix
function translate(out, a, x, y, z) {
	if (a === out) {
		out[12] = a[0] * x + a[4] * y + a[8] * z + a[12]
		out[13] = a[1] * x + a[5] * y + a[9] * z + a[13]
		out[14] = a[2] * x + a[6] * y + a[10] * z + a[14]
		out[15] = a[3] * x + a[7] * y + a[11] * z + a[15]
	} else {
		let a00, a01, a02, a03,
			a10, a11, a12, a13,
			a20, a21, a22, a23

		a00 = a[0]; a01 = a[1]; a02 = a[2]; a03 = a[3]
		a10 = a[4]; a11 = a[5]; a12 = a[6]; a13 = a[7]
		a20 = a[8]; a21 = a[9]; a22 = a[10]; a23 = a[11]

		out[0] = a00; out[1] = a01; out[2] = a02; out[3] = a03
		out[4] = a10; out[5] = a11; out[6] = a12; out[7] = a13
		out[8] = a20; out[9] = a21; out[10] = a22; out[11] = a23

		out[12] = a00 * x + a10 * y + a20 * z + a[12]
		out[13] = a01 * x + a11 * y + a21 * z + a[13]
		out[14] = a02 * x + a12 * y + a22 * z + a[14]
		out[15] = a03 * x + a13 * y + a23 * z + a[15]
	}
}

function setOrthogonal(out, l, r, b, t, near, far) {
	const lr = 1 / (l - r),
		bt = 1 / (b - t),
		nf = 1 / (near - far)
	out[0] = -2 * lr
	out[1] = 0
	out[2] = 0
	out[3] = 0
	out[4] = 0
	out[5] = -2 * bt
	out[6] = 0
	out[7] = 0
	out[8] = 0
	out[9] = 0
	out[10] = 2 * nf
	out[11] = 0
	out[12] = (l + r) * lr
	out[13] = (t + b) * bt
	out[14] = (far + near) * nf
	out[15] = 1
}

function setPerspective(out, fov, aspect, near, far) {
	const f = 1 / M.tan(fov), d = near - far
	out[0] = f / aspect
	out[1] = 0
	out[2] = 0
	out[3] = 0
	out[4] = 0
	out[5] = f
	out[6] = 0
	out[7] = 0
	out[8] = 0
	out[9] = 0
	out[10] = (far + near) / d
	out[11] = -1
	out[12] = 0
	out[13] = 0
	out[14] = (2 * far * near) / d
	out[15] = 0
}

function drawCameraModel(count, uniforms, color) {
	gl.uniform4fv(uniforms.color, color)
	gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_SHORT, 0)
}

function drawShadowModel(count) {
	gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_SHORT, 0)
}

function setCameraModel(uniforms, mm) {
	multiply(modelViewMat, lightViewMat, mm)
	gl.uniformMatrix4fv(uniforms.lightModelViewMat, false, modelViewMat)
	multiply(modelViewMat, viewMat, mm)
	gl.uniformMatrix4fv(uniforms.modelViewMat, false, modelViewMat)
}

function setShadowModel(uniforms, mm) {
	multiply(modelViewMat, lightViewMat, mm)
	gl.uniformMatrix4fv(uniforms.lightModelViewMat, false, modelViewMat)
}

function bindModel(attribs, model) {
	gl.bindBuffer(gl.ARRAY_BUFFER, model.vertices)
	gl.vertexAttribPointer(attribs.vertex, 3, gl.FLOAT, false, 0, 0)
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indicies)
}

function drawEntities(setModel, drawModel, uniforms, attribs) {
	for (let model, i = entitiesLength; i--;) {
		const e = entities[i]
		if (model != e.model) {
			model = e.model
			bindModel(attribs, model)
		}
		setModel(uniforms, e.matrix)
		drawModel(model.count, uniforms, e.color)
	}
}

function drawScreen() {
	const uniforms = screenProgram.uniforms,
		attribs = screenProgram.attribs

	gl.useProgram(screenProgram)
	gl.bindFramebuffer(gl.FRAMEBUFFER, null)
	gl.viewport(0, 0, width, height)
	gl.clearColor(skyColor[0], skyColor[1], skyColor[2], skyColor[3])
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	gl.bindBuffer(gl.ARRAY_BUFFER, screenVertexBuffer)
	gl.vertexAttribPointer(attribs.vertex, 2, gl.FLOAT, false, 0, 0)
	gl.bindBuffer(gl.ARRAY_BUFFER, screenTextureBuffer)
	gl.vertexAttribPointer(attribs.texturePos, 2, gl.FLOAT, false, 0, 0)

	gl.activeTexture(gl.TEXTURE1)
	gl.bindTexture(gl.TEXTURE_2D, offscreenTexture)
	gl.uniform1i(uniforms.offscreenTexture, 1)

	gl.enableVertexAttribArray(attribs.vertex)
	gl.enableVertexAttribArray(attribs.texturePos)
	gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
	gl.disableVertexAttribArray(attribs.vertex)
	gl.disableVertexAttribArray(attribs.texturePos)
}

function drawCameraView() {
	const uniforms = offscreenProgram.uniforms,
		attribs = offscreenProgram.attribs

	gl.useProgram(offscreenProgram)
	gl.bindFramebuffer(gl.FRAMEBUFFER, offscreenBuffer)
	gl.viewport(0, 0, offscreenWidth, offscreenHeight)
	gl.clearColor(skyColor[0], skyColor[1], skyColor[2], skyColor[3])
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	gl.uniformMatrix4fv(uniforms.projMat, false, projMat)
	gl.uniformMatrix4fv(uniforms.lightProjMat, false, lightProjMat)
	gl.uniform4fv(uniforms.sky, skyColor)
	gl.uniform1f(uniforms.far, horizon)

	gl.activeTexture(gl.TEXTURE0)
	gl.bindTexture(gl.TEXTURE_2D, shadowDepthTexture)
	gl.uniform1i(uniforms.shadowDepthTexture, 0)

	gl.enableVertexAttribArray(attribs.vertex)
	drawEntities(setCameraModel, drawCameraModel, uniforms, attribs)
	gl.disableVertexAttribArray(attribs.vertex)
}

function drawShadowMap() {
	const attribs = shadowProgram.attribs,
		uniforms = shadowProgram.uniforms

	gl.useProgram(shadowProgram)
	gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFramebuffer)
	gl.viewport(0, 0, shadowDepthTextureSize, shadowDepthTextureSize)
	gl.clearColor(0, 0, 0, 1)
	gl.clearDepth(1)
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	gl.uniformMatrix4fv(uniforms.lightProjMat, false, lightProjMat)

	gl.enableVertexAttribArray(attribs.vertex)
	drawEntities(setShadowModel, drawShadowModel, uniforms, attribs)
	gl.disableVertexAttribArray(attribs.vertex)
}

function draw() {
	drawShadowMap()
	drawCameraView()
	drawScreen()
}

function update() {
	const now = Date.now()
	for (let i = entitiesLength; i--;) {
		const e = entities[i]
		e.update && e.update(now)
	}
}

function run() {
	requestAnimationFrame(run)
	update()
	draw()
}

function raySphere(dx, dy, dz, rx, ry, rz, r) {
	const b = rx*dx + ry*dy + rz*dz,
		c = (dx*dx + dy*dy + dz*dz) - r*r,
		d = b*b - c
	// ray misses sphere
	if (d < 0) {
		return -1
	}
	const sqd = M.sqrt(d),
		ta = -b + sqd
	// ray hitting front and back
	if (d > 0) {
		if (ta >= 0) {
			return ta
		}
		const tb = -b - sqd
		if (tb >= 0) {
			return tb
		}
		return -1
	}
	// ray touching once
	return d == 0 && ta >= 0 ? ta : -1
}

function v3len(x, y, z) {
	return M.sqrt(x*x + y*y + z*z)
}

function findEntityFromViewportCoordinates(x, y) {
	// normalised device space
	const dx = (2 * x) / width - 1,
		dy = 1 - (2 * y) / height,
		dz = -1,
		dw = 1
	// camera space
	invert(findMat, projMat)
	const cx = findMat[0]*dx + findMat[1]*dy + findMat[2]*dz + findMat[3]*dw,
		cy = findMat[4]*dx + findMat[5]*dy + findMat[6]*dz + findMat[7]*dw,
		cz = -1,
		cw = 0
	// world space
	let wx = viewMat[0]*cx + viewMat[1]*cy + viewMat[2]*cz + viewMat[3]*cw,
		wy = viewMat[4]*cx + viewMat[5]*cy + viewMat[6]*cz + viewMat[7]*cw,
		wz = viewMat[8]*cx + viewMat[9]*cy + viewMat[10]*cz + viewMat[11]*cw,
		len = wx*wx + wy*wy + wz*wz
	if (len > 0) {
		len = 1 / M.sqrt(len)
	}
	wx *= len
	wy *= len
	wz *= len
	invert(findMat, viewMat)
	const ox = findMat[12],
		oy = findMat[13],
		oz = findMat[14]
	let closest = null,
		closestDist = horizon
	for (let i = entitiesLength; i--;) {
		const e = entities[i]
		if (!e.selectable) {
			continue
		}
		const em = e.matrix,
			ex = em[12],
			ey = em[13],
			ez = em[14],
			sx = v3len(em[0], em[1], em[2]),
			sy = v3len(em[4], em[5], em[6]),
			sz = v3len(em[8], em[9], em[10]),
			s = M.max(sx, M.max(sy, sz)),
			// why ex-ox and not ox-ex?
			t = raySphere(ex - ox, oy - ey, oz - ez, wx, wy, wz, s)
		if (t >= 0 && t < closestDist) {
			closestDist = t
			closest = e
		}
	}
	if (closest) {
		closest.color[0] = M.random()
		closest.color[1] = closest.color[2] = 0
	}
}

function setPointer(event, down) {
	const touches = event.touches
	if (!down) {
		pointersLength = touches ? touches.length : 0
	} else if (event.touches) {
		pointersLength = touches.length
		for (let i = pointersLength; i--;) {
			const t = touches[i]
			pointersX[i] = t.pageX
			pointersY[i] = t.pageY
		}
	} else {
		pointersLength = 1
		pointersX[0] = event.pageX
		pointersY[0] = event.pageY
	}

	if (down) {
		findEntityFromViewportCoordinates(pointersX[0], pointersY[0])

		// map to WebGL coordinates
		for (let i = pointersLength; i--;) {
			pointersX[i] = pointersX[i] * widthToGl - 1
			pointersY[i] = -(pointersY[i] * heightToGl - ymax)
		}
	}

	event.preventDefault()
	event.stopPropagation()
}

function pointerCancel(event) {
	setPointer(event, false)
}

function pointerUp(event) {
	setPointer(event, false)
}

function pointerMove(event) {
	setPointer(event, pointersLength)
}

function pointerDown(event) {
	setPointer(event, true)
}

function setKey(event, down) {
	keysDown[event.keyCode] = down
	event.stopPropagation()
}

function keyUp(event) {
	setKey(event, false)
}

function keyDown(event) {
	setKey(event, true)
}

function resize() {
	width = gl.canvas.clientWidth
	height = gl.canvas.clientHeight

	gl.canvas.width = width
	gl.canvas.height = height

	ymax = height / width
	widthToGl = 2 / width
	heightToGl = ymax * 2 / height

	setPerspective(projMat, M.PI * .125, width / height, .1, horizon)
}

function createModel(vertices, indicies) {
	const model = {count: indicies.length}

	model.vertices = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, model.vertices)
	gl.bufferData(gl.ARRAY_BUFFER, new FA(vertices), gl.STATIC_DRAW)

	model.indicies = gl.createBuffer()
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indicies)
	gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indicies),
		gl.STATIC_DRAW)

	return model
}

function createCube() {
	return createModel([
		// front
		-1, -1, 1,
		1, -1, 1,
		-1, 1, 1,
		1, 1, 1,
		// right
		1, -1, 1,
		1, -1, -1,
		1, 1, 1,
		1, 1, -1,
		// back
		1, -1, -1,
		-1, -1, -1,
		1, 1, -1,
		-1, 1, -1,
		// left
		-1, -1, -1,
		-1, -1, 1,
		-1, 1, -1,
		-1, 1, 1,
		// bottom
		-1, -1, -1,
		1, -1, -1,
		-1, -1, 1,
		1, -1, 1,
		// top
		-1, 1, 1,
		1, 1, 1,
		-1, 1, -1,
		1, 1, -1],[
		// front
		0, 1, 3,
		0, 3, 2,
		// right
		4, 5, 7,
		4, 7, 6,
		// back
		8, 9, 11,
		8, 11, 10,
		// left
		12, 13, 15,
		12, 15, 14,
		// bottom
		16, 17, 19,
		16, 19, 18,
		// top
		20, 21, 23,
		20, 23, 22])
}

function createEntities() {
	entities = []

	const cubeModel = createCube()
	let mat

	mat = new FA(idMat)
	translate(mat, mat, 0, -1, 0)
	scale(mat, mat, 3000, .01, 3000)
	entities.push({
		matrix: new FA(mat),
		model: cubeModel,
		selectable: false,
		color: [.1, .3, .7, 1]
	})

	mat = new FA(idMat)
	translate(mat, mat, 0, .1, 0)
	scale(mat, mat, 3, .1, 3)
	entities.push({
		origin: mat,
		matrix: new FA(mat),
		model: cubeModel,
		selectable: false,
		color: [.2, .4, .8, 1],
		update: function(now) {
			rotate(this.matrix, this.origin, -now * .0005, 0, 1, 0)
		}
	})

	mat = new FA(idMat)
	translate(mat, mat, 0, 1, 0)
	scale(mat, mat, .5, .5, .5)
	entities.push({
		origin: mat,
		matrix: new FA(mat),
		model: cubeModel,
		selectable: true,
		color: [1, 1, 1, 1],
		update: function(now) {
			rotate(this.matrix, this.origin, now * .001, 0, 1, 0)
		}
	})

	mat = new FA(idMat)
	translate(mat, mat, 2.5, 1.5, 0)
	rotate(mat, mat, .5, 0, 1, 0)
	scale(mat, mat, .5, .5, .5)
	entities.push({
		origin: mat,
		matrix: new FA(mat),
		model: cubeModel,
		selectable: true,
		color: [1, 0, 1, 1],
		update: function(now) {
			const m = new FA(idMat)
			rotate(m, idMat, now * .001, 0, 1, 0)
			multiply(this.matrix, m, this.origin)
		}
	})

	entitiesLength = entities.length
}

function cacheUniformLocations(program, uniforms) {
	if (program.uniforms === undefined) {
		program.uniforms = {}
	}
	for (let i = 0, l = uniforms.length; i < l; ++i) {
		const name = uniforms[i],
			loc = gl.getUniformLocation(program, name)
		if (!loc) {
			throw 'uniform "' + name + '" not found'
		}
		program.uniforms[name] = loc
	}
}

function cacheAttribLocations(program, attribs) {
	if (program.attribs === undefined) {
		program.attribs = {}
	}
	for (let i = 0, l = attribs.length; i < l; ++i) {
		const name = attribs[i],
			loc = gl.getAttribLocation(program, name)
		if (loc < 0) {
			throw 'attribute "' + name + '" not found'
		}
		program.attribs[name] = loc
	}
}

function cacheLocations(program, attribs, uniforms) {
	cacheAttribLocations(program, attribs)
	cacheUniformLocations(program, uniforms)
}

function compileShader(src, type) {
	const shader = gl.createShader(type)
	gl.shaderSource(shader, src)
	gl.compileShader(shader)
	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		throw gl.getShaderInfoLog(shader)
	}
	return shader
}

function linkProgram(vs, fs) {
	const p = gl.createProgram()
	gl.attachShader(p, vs)
	gl.attachShader(p, fs)
	gl.linkProgram(p)
	if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
		throw gl.getProgramInfoLog(p)
	}
	return p
}

function buildProgram(vertexSource, fragmentSource) {
	return linkProgram(
		compileShader(vertexSource, gl.VERTEX_SHADER),
		compileShader(fragmentSource, gl.FRAGMENT_SHADER))
}

function createPrograms() {
	shadowProgram = buildProgram(
		D.getElementById('LightVertexShader').textContent,
		D.getElementById('LightFragmentShader').textContent)
	cacheLocations(shadowProgram, ['vertex'],
		['lightProjMat', 'lightModelViewMat'])

	offscreenProgram = buildProgram(
		D.getElementById('OffscreenVertexShader').textContent,
		D.getElementById('OffscreenFragmentShader').textContent)
	cacheLocations(offscreenProgram, ['vertex'], [
		'projMat', 'modelViewMat', 'lightProjMat', 'lightModelViewMat',
		'far', 'sky', 'color', 'shadowDepthTexture'])

	screenProgram = buildProgram(
		D.getElementById('ScreenVertexShader').textContent,
		D.getElementById('ScreenFragmentShader').textContent)
	cacheLocations(screenProgram, ['vertex', 'texturePos'],
		['offscreenTexture'])
}

function createScreenBuffer() {
	screenVertexBuffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, screenVertexBuffer)
	gl.bufferData(gl.ARRAY_BUFFER,
		new FA([
			-1, 1,
			-1, -1,
			1, 1,
			1, -1]),
		gl.STATIC_DRAW)

	screenTextureBuffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, screenTextureBuffer)
	gl.bufferData(gl.ARRAY_BUFFER,
		new FA([
			1, 1,
			1, 0,
			0, 1,
			0, 0,
		]),
		gl.STATIC_DRAW)
}

function createOffscreenBuffer() {
	offscreenTexture = gl.createTexture()
	gl.bindTexture(gl.TEXTURE_2D, offscreenTexture)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, offscreenWidth,
		offscreenHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)

	const renderBuffer = gl.createRenderbuffer()
	gl.bindRenderbuffer(gl.RENDERBUFFER, renderBuffer)
	gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16,
		offscreenWidth, offscreenHeight)

	offscreenBuffer = gl.createFramebuffer()
	gl.bindFramebuffer(gl.FRAMEBUFFER, offscreenBuffer)
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D, offscreenTexture, 0)
	gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
		gl.RENDERBUFFER, renderBuffer)

	gl.bindTexture(gl.TEXTURE_2D, null)
	gl.bindRenderbuffer(gl.RENDERBUFFER, null)
	gl.bindFramebuffer(gl.FRAMEBUFFER, null)
}

function createShadowBuffer() {
	shadowDepthTexture = gl.createTexture()
	gl.bindTexture(gl.TEXTURE_2D, shadowDepthTexture)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, shadowDepthTextureSize,
		shadowDepthTextureSize, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)

	const renderBuffer = gl.createRenderbuffer()
	gl.bindRenderbuffer(gl.RENDERBUFFER, renderBuffer)
	gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16,
		shadowDepthTextureSize, shadowDepthTextureSize)

	shadowFramebuffer = gl.createFramebuffer()
	gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFramebuffer)
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D, shadowDepthTexture, 0)
	gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
		gl.RENDERBUFFER, renderBuffer)

	gl.bindTexture(gl.TEXTURE_2D, null)
	gl.bindRenderbuffer(gl.RENDERBUFFER, null)
	gl.bindFramebuffer(gl.FRAMEBUFFER, null)
}

function setLight() {
	setOrthogonal(lightProjMat, -10, 10, -10, 10, -20, 40)
	translate(lightViewMat, idMat, 0, 0, -35)
	rotate(lightViewMat, lightViewMat, M.PI2, 1, 0, 0)
}

function setCamera() {
	translate(viewMat, idMat, 0, 0, -10)
	rotate(viewMat, viewMat, .9, 1, 0, 0)
}

function init() {
	const canvas = D.getElementById('Canvas')
	gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')

	setCamera()
	setLight()

	createShadowBuffer()
	createOffscreenBuffer()
	createScreenBuffer()
	createPrograms()
	createEntities()

	gl.enable(gl.DEPTH_TEST)
	gl.enable(gl.BLEND)
	gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

	W.onresize = resize
	resize()

	D.onkeydown = keyDown
	D.onkeyup = keyUp

	D.onmousedown = pointerDown
	D.onmousemove = pointerMove
	D.onmouseup = pointerUp
	D.onmouseout = pointerCancel

	if ('ontouchstart' in D) {
		D.ontouchstart = pointerDown
		D.ontouchmove = pointerMove
		D.ontouchend = pointerUp
		D.ontouchleave = pointerCancel
		D.ontouchcancel = pointerCancel

		// prevent pinch/zoom on iOS 11
		D.addEventListener('gesturestart', function(event) {
			event.preventDefault()
		}, false)
		D.addEventListener('gesturechange', function(event) {
			event.preventDefault()
		}, false)
		D.addEventListener('gestureend', function(event) {
			event.preventDefault()
		}, false)
	}

	run()
}

W.onload = init

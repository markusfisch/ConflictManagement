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
	staticLightViewMat = new FA(16),
	lightProjMat = new FA(idMat),
	lightViewMat = new FA(idMat),
	lightDirection = [0, 0, 0],
	camPos = [0, 9, 7],
	spot = [0, 0, 0],
	skyColor = [.06, .06, .06, 1],
	offscreenWidth = 256,
	offscreenHeight = 256,
	shadowTextureSize = 1024

let gl,
	shadowBuffer,
	shadowTexture,
	shadowProgram,
	offscreenBuffer,
	offscreenTexture,
	offscreenProgram,
	screenBuffer,
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
	keysDown = [],
	drag = {},
	marker

M.PI2 = M.PI2 || M.PI / 2
M.TAU = M.TAU || M.PI * 2

function nop() {
	// it's faster to run an empty function than using a conditional
	// statement thanks to branch prediction
}

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

	d = 1 / d

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

	if (M.abs(len) < .000001) {
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

// from https://github.com/toji/gl-matrix
function transpose(out, a) {
	if (out === a) {
		const a01 = a[1], a02 = a[2], a03 = a[3],
			a12 = a[6], a13 = a[7], a23 = a[11]

		out[1] = a[4]
		out[2] = a[8]
		out[3] = a[12]
		out[4] = a01
		out[6] = a[9]
		out[7] = a[13]
		out[8] = a02
		out[9] = a12
		out[11] = a[14]
		out[12] = a03
		out[13] = a13
		out[14] = a23
	} else {
		out[0] = a[0]
		out[1] = a[4]
		out[2] = a[8]
		out[3] = a[12]
		out[4] = a[1]
		out[5] = a[5]
		out[6] = a[9]
		out[7] = a[13]
		out[8] = a[2]
		out[9] = a[6]
		out[10] = a[10]
		out[11] = a[14]
		out[12] = a[3]
		out[13] = a[7]
		out[14] = a[11]
		out[15] = a[15]
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

function drawShadowModel(count) {
	gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_SHORT, 0)
}

function drawCameraModel(count, uniforms, color) {
	gl.uniform4fv(uniforms.color, color)
	drawShadowModel(count)
}

function setCameraModel(uniforms, mm) {
	gl.uniformMatrix4fv(uniforms.modelViewMat, false, modelViewMat)
}

function drawEntities(setModel, drawModel, uniforms, attribs) {
	for (let i = entitiesLength; i--;) {
		const e = entities[i],
			model = e.model,
			bones = e.bones,
			mm = e.matrix

		// attribs & buffers
		gl.bindBuffer(gl.ARRAY_BUFFER, model.buffer)
		gl.vertexAttribPointer(attribs.vertex, 3, gl.FLOAT, false, 48, 0)
		gl.vertexAttribPointer(attribs.normal, 3, gl.FLOAT, false, 48, 12)
		gl.vertexAttribPointer(attribs.uv, 2, gl.FLOAT, false, 48, 24)
		gl.vertexAttribPointer(attribs.boneIndex, 2, gl.FLOAT, false, 48, 32)
		gl.vertexAttribPointer(attribs.boneWeight, 2, gl.FLOAT, false, 48, 40)
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indicies)

		// uniforms
		multiply(modelViewMat, lightViewMat, mm)
		gl.uniformMatrix4fv(uniforms.lightModelViewMat, false, modelViewMat)

		multiply(modelViewMat, viewMat, mm)
		setModel(uniforms, mm)
		// the model matrix needs to be inverted and transposed to
		// scale the normals correctly
		invert(modelViewMat, mm)
		transpose(modelViewMat, modelViewMat)
		gl.uniformMatrix4fv(uniforms.normalMat, false, modelViewMat)
		gl.uniformMatrix4fv(uniforms['bones[0]'], false, bones[0])
		gl.uniformMatrix4fv(uniforms['bones[1]'], false, bones[1])

		drawModel(model.count, uniforms, e.color)
	}
}

function initFrame(buffer, w, h) {
	gl.bindFramebuffer(gl.FRAMEBUFFER, buffer)
	gl.viewport(0, 0, w, h)
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
}

function initCameraFrame(buffer, w, h) {
	initFrame(buffer, w, h)
	gl.clearColor(skyColor[0], skyColor[1], skyColor[2], skyColor[3])
}

function drawScreen() {
	initCameraFrame(null, width, height)

	gl.useProgram(screenProgram)
	const uniforms = screenProgram.uniforms,
		attribs = screenProgram.attribs

	gl.bindBuffer(gl.ARRAY_BUFFER, screenBuffer)
	gl.vertexAttribPointer(attribs.vertex, 2, gl.FLOAT, false, 16, 0)
	gl.vertexAttribPointer(attribs.uv, 2, gl.FLOAT, false, 16, 8)

	gl.activeTexture(gl.TEXTURE1)
	gl.bindTexture(gl.TEXTURE_2D, offscreenTexture)
	gl.uniform1i(uniforms.offscreenTexture, 1)

	gl.enableVertexAttribArray(attribs.vertex)
	gl.enableVertexAttribArray(attribs.uv)
	gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
	gl.disableVertexAttribArray(attribs.vertex)
	gl.disableVertexAttribArray(attribs.uv)
}

function drawCameraView() {
	initCameraFrame(offscreenBuffer, offscreenWidth, offscreenHeight)

	gl.useProgram(offscreenProgram)
	const uniforms = offscreenProgram.uniforms,
		attribs = offscreenProgram.attribs

	gl.uniformMatrix4fv(uniforms.projMat, false, projMat)
	gl.uniformMatrix4fv(uniforms.lightProjMat, false, lightProjMat)
	gl.uniform3fv(uniforms.lightDirection, lightDirection)
	gl.uniform4fv(uniforms.sky, skyColor)
	gl.uniform1f(uniforms.far, horizon)

	gl.activeTexture(gl.TEXTURE0)
	gl.bindTexture(gl.TEXTURE_2D, shadowTexture)
	gl.uniform1i(uniforms.shadowTexture, 0)

	gl.enableVertexAttribArray(attribs.vertex)
	gl.enableVertexAttribArray(attribs.normal)
	gl.enableVertexAttribArray(attribs.uv)
	gl.enableVertexAttribArray(attribs.boneIndex)
	gl.enableVertexAttribArray(attribs.boneWeight)
	drawEntities(setCameraModel, drawCameraModel, uniforms, attribs)
	gl.disableVertexAttribArray(attribs.vertex)
	gl.disableVertexAttribArray(attribs.normal)
	gl.disableVertexAttribArray(attribs.uv)
	gl.disableVertexAttribArray(attribs.boneIndex)
	gl.disableVertexAttribArray(attribs.boneWeight)
}

function drawShadowMap() {
	initFrame(shadowBuffer, shadowTextureSize, shadowTextureSize)
	gl.clearColor(0, 0, 0, 1)

	gl.useProgram(shadowProgram)
	const attribs = shadowProgram.attribs,
		uniforms = shadowProgram.uniforms

	gl.uniformMatrix4fv(uniforms.lightProjMat, false, lightProjMat)
	gl.uniform3fv(uniforms.lightDirection, lightDirection)

	gl.enableVertexAttribArray(attribs.vertex)
	gl.enableVertexAttribArray(attribs.normal)
	gl.enableVertexAttribArray(attribs.uv)
	gl.enableVertexAttribArray(attribs.boneIndex)
	gl.enableVertexAttribArray(attribs.boneWeight)
	drawEntities(nop, drawShadowModel, uniforms, attribs)
	gl.disableVertexAttribArray(attribs.vertex)
	gl.disableVertexAttribArray(attribs.normal)
	gl.disableVertexAttribArray(attribs.uv)
	gl.disableVertexAttribArray(attribs.boneIndex)
	gl.disableVertexAttribArray(attribs.boneWeight)
}

function update() {
	const now = Date.now()
	for (let i = entitiesLength; i--;) {
		entities[i].update(now)
	}
}

function run() {
	requestAnimationFrame(run)
	update()
	drawShadowMap()
	drawCameraView()
	drawScreen()
}

function rayGround(out, lx, ly, lz, dx, dy, dz) {
	const denom = -1*dy
	if (denom > .0001) {
		const t = -1*-ly / denom
		out[0] = lx + dx*t
		out[1] = ly + dy*t
		out[2] = lz + dz*t
		return t >= 0
	}
	return false
}

function findGroundSpot(out, nx, ny) {
	invert(findMat, projMat)
	const cx = findMat[0]*nx + findMat[4]*ny + -findMat[8] + findMat[12],
		cy = findMat[1]*nx + findMat[5]*ny + -findMat[9] + findMat[13]
	invert(findMat, viewMat)
	let x = findMat[0]*cx + findMat[4]*cy + -findMat[8],
		y = findMat[1]*cx + findMat[5]*cy + -findMat[9],
		z = findMat[2]*cx + findMat[6]*cy + -findMat[10],
		len = x*x + y*y + z*z
	if (len > 0) {
		len = 1 / M.sqrt(len)
	}
	x *= len
	y *= len
	z *= len
	return rayGround(out, -findMat[12], findMat[13], findMat[14], x, y, z)
}

function setPointer(event, down) {
	const touches = event.touches
	if (touches) {
		pointersLength = touches.length
		for (let i = pointersLength; i--;) {
			const t = touches[i]
			pointersX[i] = t.pageX
			pointersY[i] = t.pageY
		}
	} else if (!down) {
		pointersLength = 0
	} else {
		pointersLength = 1
		pointersX[0] = event.pageX
		pointersY[0] = event.pageY
	}

	// map to WebGL coordinates
	for (let i = pointersLength; i--;) {
		pointersX[i] = (2 * pointersX[i]) / width - 1
		pointersY[i] = 1 - (2 * pointersY[i]) / height
	}

	event.stopPropagation()
}

function dragCamera() {
	const dx = pointersX[0] - drag.x,
		dy = pointersY[0] - drag.y,
		d = dx*dx + dy*dy,
		f = 8
	if (d > .001) {
		lookAt(drag.cx + dx * f, drag.cz + dy * f)
		drag.dragging = true
	}
}

function stopDrag() {
	drag.dragging = false
}

function startDrag() {
	stopDrag()
	drag.x = pointersX[0]
	drag.y = pointersY[0]
	invert(findMat, viewMat)
	drag.cx = findMat[12] - camPos[0]
	drag.cz = findMat[14] - camPos[2]
}

function pointerCancel(event) {
	setPointer(event, false)
	stopDrag()
}

function pointerUp(event) {
	setPointer(event, false)
	if (pointersLength > 0) {
		startDrag()
	} else {
		if (!drag.dragging &&
				findGroundSpot(spot, pointersX[0], pointersY[0])) {
			translate(marker.origin, idMat, -spot[0], spot[1], spot[2])
		}
		stopDrag()
	}
}

function pointerMove(event) {
	setPointer(event, pointersLength)
	dragCamera()
}

function pointerDown(event) {
	setPointer(event, true)
	startDrag()
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

function calculateNormals(vertices, indicies) {
	const normals = []

	for (let i = 0, l = indicies.length; i < l;) {
		const a = indicies[i++] * 3,
			b = indicies[i++] * 3,
			c = indicies[i++] * 3,
			x1 = vertices[a],
			y1 = vertices[a + 1],
			z1 = vertices[a + 2],
			x2 = vertices[b],
			y2 = vertices[b + 1],
			z2 = vertices[b + 2],
			x3 = vertices[c],
			y3 = vertices[c + 1],
			z3 = vertices[c + 2],
			ux = x2 - x1,
			uy = y2 - y1,
			uz = z2 - z1,
			vx = x3 - x1,
			vy = y3 - y1,
			vz = z3 - z1,
			nx = uy * vz - uz * vy,
			ny = uz * vx - ux * vz,
			nz = ux * vy - uy * vx

		normals[a] = nx
		normals[a + 1] = ny
		normals[a + 2] = nz

		normals[b] = nx
		normals[b + 1] = ny
		normals[b + 2] = nz

		normals[c] = nx
		normals[c + 1] = ny
		normals[c + 2] = nz
	}

	return normals
}

function createModel(vertices, indicies, uvs, boneIndices, boneWeights) {
	const ncoordinates = vertices.length,
		vec2elements = (ncoordinates / 3) << 1,
		model = {count: indicies.length}

	uvs = uvs || new FA(vec2elements)
	boneIndices = boneIndices || new FA(vec2elements)
	if (!boneWeights) {
		boneWeights = []
		for (let i = vec2elements; i--;) {
			boneWeights.push(.5)
		}
	}

	const buffer = [],
		normals = calculateNormals(vertices, indicies)
	for (let v = 0, n = 0, i = 0, w = 0, p = 0; v < ncoordinates;) {
		buffer.push(vertices[v++])
		buffer.push(vertices[v++])
		buffer.push(vertices[v++])
		buffer.push(normals[n++])
		buffer.push(normals[n++])
		buffer.push(normals[n++])
		buffer.push(uvs[p++])
		buffer.push(uvs[p++])
		buffer.push(boneIndices[i++])
		buffer.push(boneIndices[i++])
		buffer.push(boneWeights[w++])
		buffer.push(boneWeights[w++])
	}
	model.buffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, model.buffer)
	gl.bufferData(gl.ARRAY_BUFFER, new FA(buffer), gl.STATIC_DRAW)

	model.indicies = gl.createBuffer()
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indicies)
	gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indicies),
		gl.STATIC_DRAW)

	return model
}

function createTestModel() {
	return createModel([
		0,.55,.08,
		.18,.75,-.16,
		.23,.75,.13,
		0,1.14,.18,
		-.23,.75,.13,
		-.18,.75,-.16,
		.29,1.11,-.12,
		.23,1.16,.12,
		-.23,1.16,.12,
		-.29,1.11,-.12,
		0,1.25,-.17,
		.49,0,0,
		-.49,0,0,
		.63,.79,.00,
		-.22,1.30,-.01,
		.22,1.30,-.01,
		-.63,.79,.00,
		0,1.64,.06
	],[
		1,11,0,
		1,0,5,
		0,2,3,
		0,3,4,
		5,12,4,
		1,5,10,
		2,1,6,
		3,2,7,
		4,3,8,
		5,4,9,
		1,10,6,
		2,6,7,
		13,15,7,
		4,8,9,
		5,9,10,
		9,8,16,
		13,6,15,
		14,9,16,
		6,13,7,
		0,12,5,
		2,11,1,
		0,11,2,
		0,4,12,
		14,16,8,
		8,3,14,
		3,7,15,
		10,9,14,
		6,10,15,
		14,17,10,
		15,10,17,
		14,3,15,
		17,14,15
	])
}

function createPlane() {
	return createModel([
		-1, 1, 1,
		1, 1, 1,
		-1, 1, -1,
		1, 1, -1
	],[
		0, 1, 3,
		0, 3, 2
	],[
		1, 1,
		1, 0,
		0, 1,
		0, 0
	])
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
		1, 1, -1
	],[
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
		20, 23, 22
	],null,[
		// front
		0, 0,
		0, 0,
		0, 0,
		0, 1,
		// right
		0, 0,
		0, 0,
		0, 1,
		0, 0,
		// back
		0, 0,
		0, 0,
		0, 0,
		0, 0,
		// left
		0, 0,
		0, 0,
		0, 0,
		0, 0,
		// bottom
		0, 0,
		0, 0,
		0, 0,
		0, 0,
		// top
		0, 0,
		0, 1,
		0, 0,
		0, 0
	])
}

function createEntities() {
	entities = []

	const cubeModel = createCube()
	let mat

	mat = new FA(idMat)
	rotate(mat, mat, .16, 0, 1, 0)
	scale(mat, mat, 30, .1, 30)
	entities.push({
		matrix: new FA(mat),
		model: createPlane(),
		color: [.3, .3, .3, 1]
	})

	mat = new FA(idMat)
	translate(mat, mat, 0, 1, 0)
	scale(mat, mat, 3, .1, 3)
	entities.push({
		origin: mat,
		matrix: new FA(mat),
		model: cubeModel,
		color: [.2, .4, .8, 1],
		update: function(now) {
			rotate(this.matrix, this.origin, -now * .0005, 0, 1, 0)
		}
	})

	mat = new FA(idMat)
	translate(mat, mat, 0, 2, 0)
	scale(mat, mat, .5, .5, .5)
	entities.push({
		origin: mat,
		matrix: new FA(mat),
		model: cubeModel,
		color: [1, 1, 1, 1],
		update: function(now) {
			translate(this.matrix, this.origin, 0, 1 + M.sin(now * .001) * 2, 0)
			rotate(this.matrix, this.matrix, now * .001, 1, 1, 0)
		}
	})

	mat = new FA(idMat)
	translate(mat, mat, -3, 2.5, 0)
	scale(mat, mat, .5, .5, .5)
	entities.push({
		origin: mat,
		matrix: new FA(mat),
		model: cubeModel,
		bones: [new FA(idMat), new FA(idMat)],
		color: [0, 1, 1, 1],
		update: function(now) {
			const t = M.abs(M.sin(now * .0005)) * 3
			translate(this.bones[1], idMat, t, 0, t)
		}
	})

	mat = new FA(idMat)
	translate(mat, mat, 3.5, 2.5, 0)
	rotate(mat, mat, .5, 0, 1, 0)
	scale(mat, mat, .5, .5, .5)
	entities.push({
		origin: mat,
		matrix: new FA(mat),
		model: cubeModel,
		color: [1, 0, 1, 1],
		update: function(now) {
			const m = new FA(idMat)
			rotate(m, idMat, now * .001, 1, 1, 0)
			multiply(this.matrix, m, this.origin)
		}
	})

	mat = new FA(idMat)
	translate(mat, mat, 4, 1, 0)
	entities.push(marker = {
		origin: mat,
		matrix: new FA(mat),
		model: createTestModel(),
		color: [1, 1, 0, 1],
		update: function(now) {
			rotate(this.matrix, this.origin, now * .001, 0, 1, 0)
		}
	})

	entitiesLength = entities.length

	const defaultBones = [idMat, idMat]
	for (let i = entitiesLength; i--;) {
		const e = entities[i]
		e.update = e.update || nop
		e.bones = e.bones || defaultBones
	}
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
	const precision = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif`, lightVertexShader = `${precision}
attribute vec3 vertex;
attribute vec3 normal;
attribute vec2 uv;
attribute vec2 boneIndex;
attribute vec2 boneWeight;

uniform mat4 lightProjMat;
uniform mat4 lightModelViewMat;
uniform vec3 lightDirection;
uniform mat4 normalMat;
uniform mat4 bones[2];

varying float bias;

void main() {
	vec4 v = vec4(vertex, 1.);
	v = ((bones[int(boneIndex.x)] * v) * boneWeight.x +
			(bones[int(boneIndex.y)] * v) * boneWeight.y);
	float intensity = max(0., dot(normalize(mat3(normalMat) * normal),
		lightDirection));
	bias = .001 * (1. - intensity);
	v.w = 1.;
	gl_Position = lightProjMat * lightModelViewMat * v;
}`, lightFragmentShader = `${precision}
varying float bias;

void main() {
	const vec4 bitShift = vec4(16777216., 65536., 256., 1.);
	const vec4 bitMask = vec4(0., 1. / 256., 1. / 256., 1. / 256.);
	vec4 comp = fract((gl_FragCoord.z + bias) * bitShift);
	comp -= comp.xxyz * bitMask;
	gl_FragColor = comp;
}`, offscreenVertexShader = `${precision}
attribute vec3 vertex;
attribute vec3 normal;
attribute vec2 uv;
attribute vec2 boneIndex;
attribute vec2 boneWeight;

uniform mat4 projMat;
uniform mat4 modelViewMat;
uniform mat4 normalMat;
uniform mat4 lightModelViewMat;
uniform mat4 lightProjMat;
uniform vec3 lightDirection;
uniform mat4 bones[2];

varying float intensity;
varying float z;
varying vec4 shadowPos;
varying vec2 textureUV;

const mat4 texUnitConverter = mat4(
	.5, .0, .0, .0,
	.0, .5, .0, .0,
	.0, .0, .5, .0,
	.5, .5, .5, 1.
);

void main() {
	vec4 v = vec4(vertex, 1.);
	v = ((bones[int(boneIndex.x)] * v) * boneWeight.x +
			(bones[int(boneIndex.y)] * v) * boneWeight.y);
	gl_Position = projMat * modelViewMat * vec4(v.xyz, 1.);
	z = gl_Position.z;
	intensity = max(0., dot(normalize(mat3(normalMat) * normal),
		lightDirection));
	shadowPos = texUnitConverter * lightProjMat * lightModelViewMat * v;
	textureUV = uv;
}`, offscreenFragmentShader = `${precision}
uniform float far;
uniform vec4 sky;
uniform vec4 color;
uniform sampler2D shadowTexture;

varying float intensity;
varying float z;
varying vec4 shadowPos;
varying vec2 textureUV;

const vec4 bitShift = vec4(1. / 16777216., 1. / 65536., 1. / 256., 1.);
float decodeFloat(vec4 c) {
	return dot(c, bitShift);
}

void main() {
	float grid = 1. / 20.;
	float thresh = grid * .5;
	float ym = step(mod(textureUV.y, grid), thresh) * thresh;
	grid = step(mod(textureUV.x + ym, grid), thresh);
	float depth = decodeFloat(texture2D(shadowTexture, shadowPos.xy));
	float light = intensity > .5 ?
		.75 + step(shadowPos.z, depth) * .25 :
		1.;
	light *= max(1. - grid, .85);
	float fog = z / far;
	gl_FragColor = vec4(
		(1. - fog) * color.rgb * light + fog * sky.rgb,
		color.a);
}`, screenVertexShader = `${precision}
attribute vec2 vertex;
attribute vec2 uv;

varying vec2 textureUV;

void main() {
	gl_Position = vec4(vertex, 0., 1.);
	textureUV = uv;
}`, screenFragmentShader = `${precision}
varying vec2 textureUV;

uniform sampler2D offscreenTexture;

void main() {
	gl_FragColor = texture2D(offscreenTexture, textureUV.st);
}`

	shadowProgram = buildProgram(lightVertexShader, lightFragmentShader)
	cacheLocations(shadowProgram,
		['vertex', 'normal', 'boneIndex', 'boneWeight'],
		['normalMat',
			'lightProjMat', 'lightModelViewMat', 'lightModelViewMat',
			'bones[0]', 'bones[1]'])

	offscreenProgram = buildProgram(offscreenVertexShader,
		offscreenFragmentShader)
	cacheLocations(offscreenProgram,
		['vertex', 'normal', 'boneIndex', 'boneWeight', 'uv'],
		['projMat', 'modelViewMat', 'normalMat',
			'lightProjMat', 'lightModelViewMat', 'lightDirection',
			'bones[0]', 'bones[1]', 'far', 'sky', 'color', 'shadowTexture'])

	screenProgram = buildProgram(screenVertexShader, screenFragmentShader)
	cacheLocations(screenProgram,
		['vertex', 'uv'],
		['offscreenTexture'])
}

function createTexture(w, h) {
	const texture = gl.createTexture()
	gl.bindTexture(gl.TEXTURE_2D, texture)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
	gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA,
		gl.UNSIGNED_BYTE, null)
	return texture
}

function createFrameBuffer(w, h) {
	const rb = gl.createRenderbuffer()
	gl.bindRenderbuffer(gl.RENDERBUFFER, rb)
	gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h)

	const tx = createTexture(w, h),
		fb = gl.createFramebuffer()
	gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D, tx, 0)
	gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT,
		gl.RENDERBUFFER, rb)

	gl.bindTexture(gl.TEXTURE_2D, null)
	gl.bindRenderbuffer(gl.RENDERBUFFER, null)
	gl.bindFramebuffer(gl.FRAMEBUFFER, null)

	return {
		tx: tx,
		fb: fb
	}
}

function createScreenBuffer() {
	screenBuffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, screenBuffer)
	gl.bufferData(gl.ARRAY_BUFFER,
		new FA([
			-1, 1, 1, 1,
			-1, -1, 1, 0,
			1, 1, 0, 1,
			1, -1, 0, 0
		]),
		gl.STATIC_DRAW)
}

function createOffscreenBuffer() {
	const buf = createFrameBuffer(offscreenWidth, offscreenHeight)
	offscreenTexture = buf.tx
	offscreenBuffer = buf.fb
}

function createShadowBuffer() {
	const buf = createFrameBuffer(shadowTextureSize,
		shadowTextureSize)
	shadowTexture = buf.tx
	shadowBuffer = buf.fb
}

function clamp(v, min, max) {
	return M.max(min, M.min(max, v))
}

function lookAt(x, z) {
	x = clamp(x, -10, 10)
	z = clamp(z, -10, 10)

	translate(viewMat, idMat, x + camPos[0], camPos[1], z + camPos[2])
	rotate(viewMat, viewMat, -.9, 1, 0, 0)
	invert(viewMat, viewMat)

	translate(lightViewMat, idMat, x, 35, z)
	rotate(lightViewMat, lightViewMat, -M.PI2, 1, 0, 0)
	invert(lightViewMat, lightViewMat)
	lightDirection[0] = lightViewMat[2]
	lightDirection[1] = lightViewMat[6]
	lightDirection[2] = lightViewMat[10]
}

function init() {
	const canvas = D.getElementById('Canvas')
	gl = canvas.getContext('webgl')

	setOrthogonal(lightProjMat, -10, 10, -10, 10, -20, 40)
	lookAt(0, 0)

	createShadowBuffer()
	createOffscreenBuffer()
	createScreenBuffer()
	createPrograms()
	createEntities()

	gl.enable(gl.DEPTH_TEST)
	gl.enable(gl.BLEND)
	gl.enable(gl.CULL_FACE)
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

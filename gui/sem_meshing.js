const $ = id => document.getElementById(id);
const keys = ['image_id', 'threshold', 'dark_pores', 'pixel_size_nm', 'crop_x', 'crop_y',
  'crop_width', 'crop_height', 'min_solid_pixels', 'fill_holes_pixels', 'component',
  'simplify_px', 'boundary_size_nm', 'bulk_size_nm', 'refinement_band_nm', 'algorithm',
  'smoothing', 'ell_nm'];
let catalog, revision = 0, timer, previewData = null, meshData = null, building = false;
let meshJobId = null, meshMetadata = null, simulationId = null, animationURL = null;

function parameters() {
  const values = Object.fromEntries(keys.map(k => [k, $(k).type === 'checkbox' ? $(k).checked :
    $(k).tagName === 'SELECT' && k !== 'algorithm' ? $(k).value : Number($(k).value)]));
  values.mode = 'threshold';
  return values;
}

async function api(path, body) {
  const response = await fetch(path, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || response.statusText);
  return data;
}

async function loadMesh(path) {
  const response = await fetch(path);
  if (!response.ok) throw Error(`Mesh download failed: ${response.statusText}`);
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== 'MSH1' || buffer.byteLength < 12) throw Error('Invalid mesh response');
  const pointCount = view.getUint32(4, true), triangleCount = view.getUint32(8, true);
  const triangleOffset = 12 + pointCount * 8;
  if (triangleOffset + triangleCount * 12 !== buffer.byteLength) throw Error('Incomplete mesh response');
  return {points: new Float32Array(buffer, 12, pointCount * 2),
    triangles: new Uint32Array(buffer, triangleOffset, triangleCount * 3), pointCount, triangleCount};
}

function message(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url;
  });
}

function drawHistogram(counts = previewData?.histogram) {
  if (!counts?.length) return;
  const canvas = $('intensityHistogram'), rect = canvas.getBoundingClientRect();
  const dpr = devicePixelRatio || 1, width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  const background = ctx.createLinearGradient(0, 0, W, 0);
  background.addColorStop(0, '#07101a'); background.addColorStop(0.5, '#263744'); background.addColorStop(1, '#dcecf0');
  ctx.fillStyle = background; ctx.fillRect(0, 0, W, H);
  const values = counts.map(value => Math.log1p(value));
  const maximum = Math.max(...values, 1), barWidth = W / values.length;
  for (let i = 0; i < values.length; i++) {
    const x = i / (values.length - 1), barHeight = (H - 4) * values[i] / maximum;
    ctx.fillStyle = `hsla(${195 - 35 * x}, 85%, ${46 + 30 * x}%, .9)`;
    ctx.fillRect(i * barWidth, H - barHeight, Math.max(1, barWidth + .25), barHeight);
  }
}

class InteractiveView {
  constructor(id) {
    this.c = $(id); this.zoom = 1; this.pan = [0, 0]; this.drag = null; this.frame = 0;
    this.bindCanvas();
  }
  bindCanvas() {
    this.c.addEventListener('wheel', event => {
      event.preventDefault();
      const rect = this.c.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
      this.zoomAt(x, y, Math.max(1, Math.min(20, this.zoom * Math.exp(-event.deltaY * .0015))));
    }, {passive: false});
    this.c.onpointerdown = event => {
      this.drag = [event.clientX, event.clientY, ...this.pan];
      this.c.setPointerCapture(event.pointerId);
    };
    this.c.onpointermove = event => {
      if (!this.drag) return;
      this.pan = [this.drag[2] + event.clientX - this.drag[0], this.drag[3] + event.clientY - this.drag[1]];
      this.schedule();
    };
    this.c.onpointerup = this.c.onpointercancel = () => { this.drag = null; };
    new ResizeObserver(() => this.schedule()).observe(this.c);
  }
  dimensions() { return [this.c.clientWidth, this.c.clientHeight]; }
  resize() {
    const [W, H] = this.dimensions(), dpr = devicePixelRatio || 1;
    const width = Math.max(1, Math.round(W * dpr)), height = Math.max(1, Math.round(H * dpr));
    const changed = this.c.width !== width || this.c.height !== height;
    if (changed) { this.c.width = width; this.c.height = height; }
    return [W, H, dpr, changed];
  }
  transform(W, H) {
    if (!previewData) return [1, 0, 0];
    const w = previewData.gray.width, h = previewData.gray.height;
    const scale = Math.min((W - 40) / w, (H - 40) / h) * this.zoom;
    return [scale, (W - w * scale) / 2 + this.pan[0], (H - h * scale) / 2 + this.pan[1]];
  }
  zoomAt(x, y, nextZoom) {
    if (!previewData) { this.zoom = nextZoom; this.schedule(); return; }
    const [W, H] = this.dimensions(), [oldScale, oldX, oldY] = this.transform(W, H);
    const imageX = (x - oldX) / oldScale, imageY = (y - oldY) / oldScale;
    this.zoom = nextZoom;
    const w = previewData.gray.width, h = previewData.gray.height;
    const nextScale = Math.min((W - 40) / w, (H - 40) / h) * this.zoom;
    this.pan = [x - imageX * nextScale - (W - w * nextScale) / 2,
      y - imageY * nextScale - (H - h * nextScale) / 2];
    this.schedule();
  }
  fit() { this.zoom = 1; this.pan = [0, 0]; this.schedule(); }
  schedule() {
    if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.draw(); });
  }
}

class ImageView extends InteractiveView {
  constructor(id) { super(id); this.ctx = this.c.getContext('2d', {alpha: false}); }
  draw() {
    const [W, H, dpr] = this.resize(), ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = '#070c12'; ctx.fillRect(0, 0, W, H);
    if (!previewData) { ctx.fillStyle = '#91a6bd'; ctx.fillText('Image preview will appear here', 25, H / 2); return; }
    const [scale, x, y] = this.transform(W, H);
    ctx.save(); ctx.translate(x, y); ctx.scale(scale, scale); ctx.drawImage(previewData.gray, 0, 0);
    if ($('overlay').checked) { ctx.globalAlpha = .38; ctx.drawImage(previewData.colored, 0, 0); }
    ctx.restore(); ctx.fillStyle = '#e4edf6'; ctx.font = '11px system-ui'; ctx.fillText(`${this.zoom.toFixed(1)}× zoom`, 12, H - 12);
  }
}

function compileProgram(gl, vertexSource, fragmentSource) {
  const shader = (type, source) => {
    const value = gl.createShader(type); gl.shaderSource(value, source); gl.compileShader(value);
    if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(value));
    return value;
  };
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragmentSource)); gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
  return program;
}

class MeshView extends InteractiveView {
  constructor(id) {
    super(id); this.mesh = null; this.path = null;
    this.gl = this.c.getContext('webgl2', {alpha: false, antialias: true, powerPreference: 'high-performance'});
    this.backend = this.gl ? this.webGLLabel() : 'Canvas 2D fallback';
    if (this.gl) {
      this.c.addEventListener('webglcontextlost', event => { event.preventDefault(); this.useCanvasFallback(); }, {once: true});
      this.initializeGL();
    } else this.ctx = this.c.getContext('2d', {alpha: false});
    $('meshHint').textContent = `Conforming P1 triangles · ${this.backend}`;
  }
  webGLLabel() {
    const extension = this.gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = extension ? this.gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : '';
    return /swiftshader|software/i.test(renderer) ? 'WebGL2 software' : 'WebGL2 GPU';
  }
  useCanvasFallback() {
    if (!this.gl) return;
    this.gl = null;
    const replacement = document.createElement('canvas'); replacement.id = this.c.id;
    this.c.replaceWith(replacement); this.c = replacement; this.bindCanvas();
    this.ctx = this.c.getContext('2d', {alpha: false}); this.backend = 'Canvas 2D fallback';
    $('meshHint').textContent = `Conforming P1 triangles · ${this.backend}`;
    if (this.mesh) this.buildCanvasPath();
    this.schedule();
  }
  initializeGL() {
    const gl = this.gl;
    const baseVertex = `#version 300 es
      precision highp float; in vec2 a_position; uniform vec2 u_view; uniform float u_scale; uniform vec2 u_offset;
      void main(){vec2 p=u_offset+a_position*u_scale;gl_Position=vec4(p.x/u_view.x*2.0-1.0,1.0-p.y/u_view.y*2.0,0.0,1.0);}`;
    const textureVertex = `#version 300 es
      precision highp float; in vec2 a_position; in vec2 a_uv; out vec2 v_uv; uniform vec2 u_view; uniform float u_scale; uniform vec2 u_offset;
      void main(){vec2 p=u_offset+a_position*u_scale;gl_Position=vec4(p.x/u_view.x*2.0-1.0,1.0-p.y/u_view.y*2.0,0.0,1.0);v_uv=a_uv;}`;
    this.textureProgram = compileProgram(gl, textureVertex, `#version 300 es
      precision mediump float; in vec2 v_uv; uniform sampler2D u_texture; out vec4 color; void main(){color=texture(u_texture,v_uv);}`);
    this.meshProgram = compileProgram(gl, baseVertex, `#version 300 es
      precision mediump float; uniform vec4 u_color; out vec4 color; void main(){color=u_color;}`);
    this.quadVAO = gl.createVertexArray(); this.quadBuffer = gl.createBuffer();
    gl.bindVertexArray(this.quadVAO); gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    let location = gl.getAttribLocation(this.textureProgram, 'a_position');
    gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 16, 0);
    location = gl.getAttribLocation(this.textureProgram, 'a_uv');
    gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 16, 8);
    this.meshVAO = gl.createVertexArray(); this.vertexBuffer = gl.createBuffer();
    this.triangleBuffer = gl.createBuffer(); this.edgeBuffer = gl.createBuffer();
    gl.bindVertexArray(this.meshVAO); gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    location = gl.getAttribLocation(this.meshProgram, 'a_position');
    gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 8, 0);
    this.texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }
  setPreview() {
    if (!this.gl || !previewData) { this.schedule(); return; }
    const gl = this.gl, w = previewData.gray.width, h = previewData.gray.height;
    const quad = new Float32Array([0, 0, 0, 1, w, 0, 1, 1, 0, h, 0, 0, w, h, 1, 0]);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer); gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    gl.bindTexture(gl.TEXTURE_2D, this.texture); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, previewData.gray);
    this.schedule();
  }
  setMesh(data) {
    this.mesh = data; this.path = null;
    if (!data || !previewData) { this.schedule(); return; }
    const physicalScale = previewData.metadata.parameters.pixel_size_nm * 1e-6;
    const vertices = new Float32Array(data.points.length), imageHeight = previewData.gray.height;
    for (let i = 0; i < data.pointCount; i++) {
      vertices[2 * i] = data.points[2 * i] / physicalScale;
      vertices[2 * i + 1] = imageHeight - data.points[2 * i + 1] / physicalScale;
    }
    data.displayPoints = vertices;
    if (this.gl) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer); gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.triangleBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data.triangles, gl.STATIC_DRAW);
      const edges = new Uint32Array(data.triangles.length * 2);
      for (let i = 0, j = 0; i < data.triangles.length; i += 3) {
        const a = data.triangles[i], b = data.triangles[i + 1], c = data.triangles[i + 2];
        edges[j++] = a; edges[j++] = b; edges[j++] = b; edges[j++] = c; edges[j++] = c; edges[j++] = a;
      }
      data.edgeCount = edges.length;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.edgeBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, edges, gl.STATIC_DRAW);
    } else this.buildCanvasPath();
    this.schedule();
  }
  buildCanvasPath() {
    if (!this.mesh?.displayPoints) return;
    const data = this.mesh, vertices = data.displayPoints; this.path = new Path2D();
    for (let i = 0; i < data.triangles.length; i += 3) {
      const a = data.triangles[i] * 2, b = data.triangles[i + 1] * 2, c = data.triangles[i + 2] * 2;
      this.path.moveTo(vertices[a], vertices[a + 1]); this.path.lineTo(vertices[b], vertices[b + 1]);
      this.path.lineTo(vertices[c], vertices[c + 1]); this.path.closePath();
    }
  }
  uniforms(program, W, H, scale, x, y) {
    const gl = this.gl;
    gl.uniform2f(gl.getUniformLocation(program, 'u_view'), W, H);
    gl.uniform1f(gl.getUniformLocation(program, 'u_scale'), scale);
    gl.uniform2f(gl.getUniformLocation(program, 'u_offset'), x, y);
  }
  drawGL(W, H, dpr) {
    const gl = this.gl; gl.viewport(0, 0, Math.round(W * dpr), Math.round(H * dpr));
    gl.clearColor(.027, .047, .071, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    if (!previewData) return;
    const [scale, x, y] = this.transform(W, H);
    if ($('backdrop').checked) {
      gl.useProgram(this.textureProgram); this.uniforms(this.textureProgram, W, H, scale, x, y);
      gl.bindVertexArray(this.quadVAO); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(gl.getUniformLocation(this.textureProgram, 'u_texture'), 0); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    if (!this.mesh) return;
    gl.useProgram(this.meshProgram); this.uniforms(this.meshProgram, W, H, scale, x, y); gl.bindVertexArray(this.meshVAO);
    gl.uniform4f(gl.getUniformLocation(this.meshProgram, 'u_color'), .384, .898, .808, .10);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.triangleBuffer); gl.drawElements(gl.TRIANGLES, this.mesh.triangles.length, gl.UNSIGNED_INT, 0);
    if ($('wire').checked) {
      gl.uniform4f(gl.getUniformLocation(this.meshProgram, 'u_color'), .384, .898, .808, .82);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.edgeBuffer); gl.drawElements(gl.LINES, this.mesh.edgeCount, gl.UNSIGNED_INT, 0);
    }
  }
  drawCanvas(W, H, dpr) {
    const ctx = this.ctx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = '#070c12'; ctx.fillRect(0, 0, W, H);
    if (!previewData) return;
    const [scale, x, y] = this.transform(W, H); ctx.save(); ctx.translate(x, y); ctx.scale(scale, scale);
    if ($('backdrop').checked) ctx.drawImage(previewData.gray, 0, 0);
    if (this.path) { ctx.fillStyle = 'rgba(98,229,206,.10)'; ctx.fill(this.path); if ($('wire').checked) { ctx.strokeStyle = '#62e5ce'; ctx.lineWidth = .65 / scale; ctx.stroke(this.path); } }
    ctx.restore();
  }
  draw() {
    const [W, H, dpr] = this.resize();
    if (this.gl) this.drawGL(W, H, dpr); else this.drawCanvas(W, H, dpr);
  }
}

const imageView = new ImageView('imageCanvas'), meshView = new MeshView('meshCanvas');
new ResizeObserver(() => drawHistogram()).observe($('intensityHistogram'));

function changed() {
  revision++; $('thresholdValue').textContent = Number($('threshold').value).toFixed(3); drawHistogram();
  $('downloads').replaceChildren();
  if (meshData) {
    message('Settings changed. Build a new mesh to update the result.'); meshData = null; meshView.setMesh(null);
    meshJobId = null; meshMetadata = null; updateSimulationAvailability();
    ['nodes', 'cells', 'dofs', 'quality'].forEach(k => $(k).textContent = '—');
  }
  clearTimeout(timer); timer = setTimeout(preview, 300);
}

async function preview() {
  const currentRevision = revision, p = parameters();
  try {
    const data = await api('/api/preview', p);
    const [gray, mask] = await Promise.all([loadImage(data.image), loadImage(data.mask)]);
    if (currentRevision !== revision) return;
    const colored = document.createElement('canvas'); colored.width = mask.width; colored.height = mask.height;
    const ctx = colored.getContext('2d'); ctx.drawImage(mask, 0, 0);
    const pixels = ctx.getImageData(0, 0, mask.width, mask.height);
    for (let i = 0; i < pixels.data.length; i += 4) {
      const solid = pixels.data[i] > 0; pixels.data[i] = 98; pixels.data[i + 1] = 229;
      pixels.data[i + 2] = 206; pixels.data[i + 3] = solid ? 255 : 0;
    }
    ctx.putImageData(pixels, 0, 0);
    previewData = {gray, colored, histogram: data.intensity_histogram, metadata: data.metadata};
    meshView.setPreview(); drawHistogram(); imageView.schedule(); meshView.schedule();
    $('pixelBadge').textContent = `${p.pixel_size_nm} nm / pixel · ${p.pixel_size_nm === 10 ? 'published scale' : 'custom scale'}`;
    const selected = catalog.images.find(image => image.id === p.image_id);
    $('activeImage').textContent = `${p.image_id} · slice ${selected.slice_index}`;
    $('dimensions').textContent = `${(data.metadata.width * 1000).toFixed(2)} × ${(data.metadata.height * 1000).toFixed(2)} µm · ${(data.metadata.porosity * 100).toFixed(1)}% pore area`;
    $('imageInfo').textContent = `${catalog.images.length} local images · ${selected.width} × ${selected.height} px`;
    if (!building && !meshData) message(`Threshold preview ready. ${data.metadata.warnings.join('\n')}`);
  } catch (error) { if (currentRevision === revision) message(error.message, true); }
}

async function build() {
  if (building) return;
  building = true; $('fetchSlice').disabled = true; $('build').disabled = true; $('build').textContent = 'Building…';
  const currentRevision = revision, p = parameters();
  meshJobId = null; meshMetadata = null; updateSimulationAvailability();
  message('Building calibrated contours and a conforming mesh…');
  try {
    const job = await api('/api/build', p); let state;
    do { await new Promise(resolve => setTimeout(resolve, 700)); state = await api(`/api/job?id=${job.id}`); } while (state.status === 'running');
    if (state.status === 'failed') throw Error(state.error);
    if (currentRevision !== revision) { message('Mesh finished for earlier settings. Build again for the current settings.'); return; }
    await preview(); const builtData = await loadMesh(`/api/mesh?id=${job.id}`);
    if (currentRevision !== revision) { message('Mesh finished for earlier settings. Build again for the current settings.'); return; }
    meshData = builtData; meshView.setMesh(meshData); meshView.fit();
    const metadata = state.metadata;
    meshJobId = job.id; meshMetadata = metadata; updateSimulationAvailability();
    for (const [key, value] of [['nodes', metadata.nodes], ['cells', metadata.cells], ['dofs', metadata.total_dofs]]) $(key).textContent = value.toLocaleString();
    $('quality').textContent = metadata.minimum_quality.toFixed(3);
    message(`Mesh built in ${metadata.mesh_seconds.toFixed(2)} s · ${metadata.boundary_curves} boundary curves · ${meshView.backend}\nContour area difference: ${(100 * metadata.contour_area_relative_error).toFixed(3)}% · ℓ / interior size: ${(p.ell_nm / p.bulk_size_nm).toFixed(2)}\n${metadata.ready_for_fracture ? 'Connected horizontal load path: ready for fracture export.' : 'Geometry is meshable; fracture export needs a single connected load path.'}\n${metadata.warnings.join('\n')}`);
    $('downloads').replaceChildren();
    const files = [['mesh/specimen.msh', 'Download mesh'], ['parameters.json', 'Save mesh settings'], ['mesh/geometry.json', 'Geometry report'], ['mesh/solid_mask.png', 'Solid mask'], ...(metadata.ready_for_fracture ? [['fracture_config.json', 'Fracture configuration']] : [])];
    for (const [file, label] of files) { const link = document.createElement('a'); link.href = `/api/download?id=${job.id}&file=${encodeURIComponent(file)}`; link.download = file.split('/').pop(); link.textContent = label; $('downloads').append(link); }
  } catch (error) { message(error.message, true); }
  finally { building = false; $('fetchSlice').disabled = false; $('build').disabled = false; $('build').textContent = 'Build mesh'; meshView.schedule(); }
}

function updateSimulationAvailability() {
  const button = $('simulate'), ready = Boolean(meshJobId && meshMetadata?.ready_for_fracture);
  button.disabled = !ready || Boolean(simulationId);
  button.textContent = ready ? (simulationId ? 'Simulation running…' : 'Simulate fracture') : 'Build a fracture-ready mesh first';
  if (!simulationId) $('simulationStatus').textContent = ready ?
    `Ready: this exact ${meshMetadata.nodes.toLocaleString()}-node mesh will be imported by DOLFINx.` :
    'The solid must form one connected load path between the left and right grips.';
}

async function simulateFracture() {
  if (!meshJobId || !meshMetadata?.ready_for_fracture || simulationId) return;
  if (animationURL) { URL.revokeObjectURL(animationURL); animationURL = null; }
  $('simulationVideo').pause(); $('simulationVideo').removeAttribute('src'); $('simulationVideo').classList.remove('ready');
  $('downloadAnimation').classList.remove('ready'); $('simulationProgress').value = 0;
  try {
    const job = await api('/api/simulate', {mesh_id: meshJobId, backend: $('simulationBackend').value,
      deformation_scale: Number($('deformationScale').value), fps: Number($('simulationFPS').value)});
    simulationId = job.id; updateSimulationAvailability();
    $('simulationStatus').textContent = `Starting ${job.backend.toUpperCase()} fracture solve on the exact generated mesh…`;
    await pollSimulation(job.id);
  } catch (error) {
    simulationId = null; updateSimulationAvailability(); $('simulationStatus').textContent = error.message;
  }
}

async function pollSimulation(id) {
  while (simulationId === id) {
    const job = await api(`/api/job?id=${id}`); $('simulationProgress').value = job.progress || 0;
    if (job.status === 'running') {
      const percent = Math.round(100 * (job.progress || 0));
      $('simulationStatus').textContent = job.accepted_steps ?
        `Solving: ${job.accepted_steps} accepted steps · separation ${(job.separation * 1000).toFixed(3)} µm · max damage ${(job.damage_max || 0).toFixed(3)} · ${percent}%` :
        'Importing the exact Gmsh mesh and compiling the fracture forms…';
    } else if (job.status === 'rendering') {
      $('simulationStatus').textContent = `${job.stopping_reason ? 'Solver stopped early.' : 'Solve complete.'} Rendering ${job.accepted_steps} accepted load steps into the animation…`;
    } else if (job.status === 'complete' || job.status === 'partial') {
      const response = await fetch(`/api/simulation/video?id=${id}`);
      if (!response.ok) throw Error(`Animation download failed: ${response.statusText}`);
      animationURL = URL.createObjectURL(await response.blob());
      const video = $('simulationVideo'); video.src = animationURL; video.classList.add('ready');
      const download = $('downloadAnimation'); download.href = animationURL; download.classList.add('ready');
      $('simulationProgress').value = 1;
      simulationId = null; updateSimulationAvailability();
      const timing = job.summary.compute_seconds == null ? '' : ` · ${job.summary.compute_seconds.toFixed(2)} s solve time`;
      $('simulationStatus').textContent = `${job.status === 'partial' ? 'Partial animation' : 'Complete'}: ${job.summary.dofs.total.toLocaleString()} DOFs · ${job.accepted_steps} accepted steps · ${job.frame_count} frames · max damage ${job.damage_max.toFixed(3)}${timing}.${job.stopping_reason ? ' Stopped: ' + job.stopping_reason : ''}`;
      video.play().catch(() => {}); return;
    } else if (job.status === 'failed') {
      throw Error(job.error || 'Fracture simulation failed');
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

async function fetchSlice() {
  const button = $('fetchSlice'); button.disabled = true; $('build').disabled = true; button.textContent = 'Fetching…';
  message('Fetching one image/mask pair from Zenodo…');
  try {
    const sample = Number($('image_id').value.split('_')[0].replace('HPC', ''));
    const job = await api('/api/fetch', {sample, index: Number($('fetchIndex').value)}); let state;
    do { await new Promise(resolve => setTimeout(resolve, 800)); state = await api(`/api/job?id=${job.id}`); } while (state.status === 'running');
    if (state.status === 'failed') throw Error(state.error);
    await init(); $('image_id').value = state.image_id; imageView.fit(); meshView.fit(); changed(); await preview();
  } catch (error) { message(error.message, true); }
  finally { button.disabled = false; $('build').disabled = building; button.textContent = 'Fetch slice'; }
}

async function init() {
  try {
    const previous = catalog ? parameters() : null; catalog = await api('/api/catalog'); const selected = $('image_id').value;
    $('image_id').replaceChildren();
    for (const image of catalog.images) { const option = document.createElement('option'); option.value = image.id; option.textContent = `${image.sample} / slice ${String(image.slice_index).padStart(4, '0')}`; $('image_id').append(option); }
    for (const key of keys) {
      const element = $(key), value = previous?.[key] ?? (key === 'image_id' ? (selected || catalog.defaults[key]) : catalog.defaults[key]);
      if (element.type === 'checkbox') element.checked = value; else element.value = value;
    }
    if (!$('image_id').value) $('image_id').selectedIndex = 0;
    changed(); await preview();
    const savedSimulation = new URLSearchParams(location.search).get('simulation');
    if (!simulationId && savedSimulation && /^[a-f0-9]{32}$/.test(savedSimulation) && !animationURL) {
      simulationId = savedSimulation;
      updateSimulationAvailability();
      pollSimulation(savedSimulation).catch(error => {
        simulationId = null; updateSimulationAvailability(); $('simulationStatus').textContent = error.message;
      });
    }
  } catch (error) { message(`Dataset could not be opened: ${error.message}`, true); }
}

keys.forEach(key => $(key).addEventListener('input', changed));
$('preview').onclick = preview; $('build').onclick = build; $('reload').onclick = init; $('fetchSlice').onclick = fetchSlice;
$('simulate').onclick = simulateFracture;
for (const [id, delta] of [['previous', -1], ['next', 1]]) $(id).onclick = () => {
  $('image_id').selectedIndex = ($('image_id').selectedIndex + delta + catalog.images.length) % catalog.images.length;
  imageView.fit(); meshView.fit(); changed();
};
$('fitImage').onclick = () => imageView.fit(); $('fitMesh').onclick = () => meshView.fit();
['overlay', 'wire', 'backdrop'].forEach(key => $(key).onchange = () => { imageView.schedule(); meshView.schedule(); });
init();

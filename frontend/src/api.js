// Thin API client for the FastAPI backend. In dev, Vite proxies /api and /media
// to :8000 (see vite.config.js); in prod the backend serves both.
import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function getCameras() {
  const { data } = await api.get('/cameras')
  return data
}

export async function getVideos() {
  const { data } = await api.get('/videos')
  return data
}

export async function getLibrary() {
  const { data } = await api.get('/library')
  return data
}

export async function ingestAll() {
  const { data } = await api.post('/ingest/all')
  return data
}

export async function getIngestJob(jobId) {
  const { data } = await api.get(`/ingest/job/${jobId}`)
  return data
}

// Upload a video from the user's device, save it server-side, and start
// ingestion. onProgress reports upload % (0-100) while bytes transfer.
export async function uploadVideo({ file, cameraId, onProgress }) {
  const form = new FormData()
  form.append('file', file)
  if (cameraId) form.append('camera_id', cameraId)
  const { data } = await api.post('/ingest/upload', form, {
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    },
  })
  return data
}

export async function stopIngest() {
  const { data } = await api.post('/ingest/stop')
  return data
}

// Permanently delete a video: file on disk + DB rows + FAISS vectors + crops.
export async function deleteVideo(filename) {
  const { data } = await api.post('/videos/delete', { filename })
  return data
}

export async function addCamera({ cameraId, name, location, lat, lon }) {
  const num = (v) => (v === '' || v == null ? null : Number(v))
  const { data } = await api.post('/cameras', {
    camera_id: cameraId, name: name || null, location: location || null,
    lat: num(lat), lon: num(lon),
  })
  return data
}

export async function searchText({ query, language = 'en', topK = 60,
                                   includeScenes = false, filters = {} }) {
  const { data } = await api.post('/search/text', {
    query, language, top_k: topK, include_scenes: includeScenes, filters,
  })
  return data
}

// Describe-and-filter search: parses the plain-language query into structured
// constraints, filters exactly on them, and annotates each result with matches.
export async function describeSearch({ query, topK = 60, filters = {} }) {
  const { data } = await api.post('/search/describe', {
    query, top_k: topK, filters,
  })
  return data
}

export async function searchImage({ file, topK = 60, useReid = false }) {
  const form = new FormData()
  form.append('file', file)
  form.append('top_k', String(topK))
  form.append('use_reid', String(useReid))
  const { data } = await api.post('/search/image', form)
  return data
}

export async function searchPlate({ plate, filters = {} }) {
  const { data } = await api.post('/search/plate', { plate, filters })
  return data
}

export async function trackDetection(detectionId) {
  const { data } = await api.get(`/track/${detectionId}`)
  return data
}

export async function getAudit(limit = 50) {
  const { data } = await api.get(`/audit?limit=${limit}`)
  return data
}

export async function createExport({ detectionIds, caseNumber, officer, notes }) {
  const { data } = await api.post('/export', {
    detection_ids: detectionIds, case_number: caseNumber, officer, notes,
  })
  return data
}

export async function getExports() {
  const { data } = await api.get('/exports')
  return data
}

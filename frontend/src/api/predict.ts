
import { apiClient } from './client'

export interface PredictResponse {
  prediction: string
  confidence: number
  confidence_level: string
  probabilities: Record<string, number>
  gradcam_image: string | null
  heatmap_image: string | null
  overlay_image: string | null
  report_pdf: string | null
  report_html: string | null
  report_json: string | null
  report_id: string
  model: string
  model_version: string
  device: string
  inference_time_ms: number
  timestamp: string
}

export interface APIResponse<T = unknown> {
  success: boolean
  message: string
  data: T | null
  metadata?: Record<string, unknown>
}

export async function runPrediction(
  file: File,
  onUploadProgress?: (percent: number) => void
): Promise<PredictResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const res = await apiClient.post<APIResponse<PredictResponse>>('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (e.total && onUploadProgress) {
        onUploadProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })

  if (!res.data.success || !res.data.data) {
    throw new Error(res.data.message || 'Prediction failed')
  }
  return res.data.data
}

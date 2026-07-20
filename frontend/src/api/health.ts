
import { apiClient } from './client'

export interface HealthData {
  status: string
  version: string
  uptime_seconds: number
  device: string
  model_loaded: boolean
}

export interface ModelInfo {
  architecture: string
  version: string
  framework: string
  device: string
  classes: string[]
}

export async function fetchHealth(): Promise<HealthData> {
  const res = await apiClient.get('/health')
  return res.data.data
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await apiClient.get('/model')
  return res.data.data
}

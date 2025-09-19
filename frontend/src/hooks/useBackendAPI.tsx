import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

export interface APIResponse<T> {
  data: T | null
  loading: boolean
  error: string | null
}

export interface ProgressData {
  task_id: string
  status: string
  progress: number
  phase: string
  message: string
  error?: string
}

export interface LogEntry {
  timestamp: string
  level: string
  message: string
  phase?: string
}

export interface FileData {
  path: string
  name: string
  size: number
  modified: number
  extension: string
  type: string
}

// Custom hook for backend API integration
export const useBackendAPI = () => {
  const [isConnected, setIsConnected] = useState<boolean>(false)

  // Test backend connection
  useEffect(() => {
    const testConnection = async () => {
      try {
        await axios.get('/api/recent')
        setIsConnected(true)
      } catch (error) {
        console.error('Backend connection failed:', error)
        setIsConnected(false)
      }
    }

    testConnection()
    const interval = setInterval(testConnection, 30000) // Check every 30s
    return () => clearInterval(interval)
  }, [])

  // Get real-time progress
  const useProgress = (taskId: string | null): APIResponse<ProgressData> => {
    const [data, setData] = useState<ProgressData | null>(null)
    const [loading, setLoading] = useState<boolean>(false)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
      if (!taskId || !isConnected) return

      setLoading(true)
      const fetchProgress = async () => {
        try {
          const response = await axios.get(`/api/progress/${taskId}`)
          setData(response.data)
          setError(null)
        } catch (err: any) {
          setError(err.message)
          setData(null)
        } finally {
          setLoading(false)
        }
      }

      fetchProgress()
      const interval = setInterval(fetchProgress, 1000)
      return () => clearInterval(interval)
    }, [taskId, isConnected])

    return { data, loading, error }
  }

  // Get real-time logs
  const useLogs = (taskId: string | null): APIResponse<LogEntry[]> => {
    const [data, setData] = useState<LogEntry[] | null>(null)
    const [loading, setLoading] = useState<boolean>(false)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
      if (!taskId || !isConnected) return

      setLoading(true)
      const fetchLogs = async () => {
        try {
          const response = await axios.get(`/api/logs/${taskId}`)
          setData(response.data.logs || [])
          setError(null)
        } catch (err: any) {
          setError(err.message)
          setData(null)
        } finally {
          setLoading(false)
        }
      }

      fetchLogs()
      const interval = setInterval(fetchLogs, 2000)
      return () => clearInterval(interval)
    }, [taskId, isConnected])

    return { data, loading, error }
  }

  // Get file list
  const useFiles = (taskId: string | null): APIResponse<FileData[]> => {
    const [data, setData] = useState<FileData[] | null>(null)
    const [loading, setLoading] = useState<boolean>(false)
    const [error, setError] = useState<string | null>(null)

    const fetchFiles = useCallback(async () => {
      if (!taskId || !isConnected) return

      setLoading(true)
      try {
        const response = await axios.get(`/api/files/${taskId}`)
        setData(response.data.files || [])
        setError(null)
      } catch (err: any) {
        setError(err.message)
        setData(null)
      } finally {
        setLoading(false)
      }
    }, [taskId, isConnected])

    useEffect(() => {
      fetchFiles()
    }, [fetchFiles])

    return { data, loading, error, refetch: fetchFiles }
  }

  // Get file content
  const getFileContent = async (taskId: string, filePath: string): Promise<string> => {
    if (!isConnected) throw new Error('Backend not connected')
    
    try {
      const response = await axios.get(`/api/view/${taskId}?file_path=${encodeURIComponent(filePath)}`)
      return response.data.content || ''
    } catch (error) {
      console.error('Failed to fetch file content:', error)
      throw error
    }
  }

  // Download project
  const downloadProject = async (taskId: string): Promise<void> => {
    if (!isConnected) throw new Error('Backend not connected')
    
    try {
      const response = await axios.get(`/api/download/${taskId}`, {
        responseType: 'blob'
      })
      
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `project_${taskId}.zip`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download project:', error)
      throw error
    }
  }

  // Get analysis data
  const useAnalysis = (taskId: string | null): APIResponse<any> => {
    const [data, setData] = useState<any | null>(null)
    const [loading, setLoading] = useState<boolean>(false)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
      if (!taskId || !isConnected) return

      const fetchAnalysis = async () => {
        setLoading(true)
        try {
          const response = await axios.get(`/api/analysis/${taskId}`)
          setData(response.data)
          setError(null)
        } catch (err: any) {
          // Analysis might not be ready yet
          if (err.response?.status !== 404) {
            setError(err.message)
          }
          setData(null)
        } finally {
          setLoading(false)
        }
      }

      fetchAnalysis()
      const interval = setInterval(fetchAnalysis, 5000)
      return () => clearInterval(interval)
    }, [taskId, isConnected])

    return { data, loading, error }
  }

  return {
    isConnected,
    useProgress,
    useLogs,
    useFiles,
    useAnalysis,
    getFileContent,
    downloadProject
  }
}

// Connection status indicator component
export const ConnectionStatus: React.FC<{ darkMode: boolean }> = ({ darkMode }) => {
  const { isConnected } = useBackendAPI()

  return (
    <div className="fixed top-4 right-4 z-50">
      <div className={`flex items-center gap-2 px-3 py-2 rounded-lg shadow-lg ${
        isConnected 
          ? 'bg-green-500 text-white' 
          : 'bg-red-500 text-white'
      }`}>
        <div className={`w-2 h-2 rounded-full ${
          isConnected ? 'bg-green-200' : 'bg-red-200'
        }`} />
        <span className="text-sm font-medium">
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </div>
  )
}

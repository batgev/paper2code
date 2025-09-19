import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useDropzone } from 'react-dropzone'
import { 
  Upload, 
  Globe, 
  Settings, 
  Play, 
  CheckCircle, 
  XCircle, 
  Clock, 
  FileText, 
  Folder, 
  Zap,
  Brain,
  Moon,
  Sun,
  ChevronRight,
  RefreshCw,
  Cpu,
  Activity,
  Sparkles,
  TrendingUp,
  Award
} from 'lucide-react'
import axios from 'axios'
import ProgressTimeline from './components/ProgressTimeline'
import ProcessLogger from './components/ProcessLogger'
import LLMProviderSelector from './components/LLMProviderSelector'
import AnalysisVisualizer from './components/AnalysisVisualizer'
import InteractiveFileExplorer from './components/InteractiveFileExplorer'
import AdvancedPDFViewer from './components/AdvancedPDFViewer'
import LiveLLMVisualizer from './components/LiveLLMVisualizer'
import CreativeCodeEditor from './components/CreativeCodeEditor'
import ImmersiveProcessVisualization from './components/ImmersiveProcessVisualization'
import LLMStreamViewer from './components/LLMStreamViewer'
import ProfessionalProcessView from './components/ProfessionalProcessView'
import { ConnectionStatus } from './hooks/useBackendAPI'

const API_BASE = ''

type ProcessResponse = {
  success: boolean
  output_path?: string
  files?: string[]
  error?: string
  processing_time?: number
}

type RecentItem = {
  name: string
  path: string
  created: number
  modified: number
  size_mb: number
}

type Phase = {
  id: string
  name: string
  icon: React.ReactNode
  status: 'pending' | 'running' | 'completed' | 'error'
  progress: number
}

export default function App() {
  const [darkMode, setDarkMode] = useState(false)
  const [currentStep, setCurrentStep] = useState(1)
  const [mode, setMode] = useState<'comprehensive' | 'fast'>('comprehensive')
  const [segmentation, setSegmentation] = useState(true)
  const [threshold, setThreshold] = useState(50000)
  const [outputDir, setOutputDir] = useState('./output')
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<ProcessResponse | null>(null)
  const [recent, setRecent] = useState<RecentItem[]>([])
  const [taskId, setTaskId] = useState<string | null>(null)
  const [progress, setProgress] = useState<number>(0)
  const [progressMsg, setProgressMsg] = useState<string>('')
  const [toast, setToast] = useState<{type: 'success'|'error'|'info', msg: string} | null>(null)
  const [llmProvider, setLlmProvider] = useState<string>('ollama')
  const [llmModel, setLlmModel] = useState<string>('deepseek-r1:8b')
  const [apiKeys, setApiKeys] = useState<{[key: string]: string}>({})
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [showResults, setShowResults] = useState(false)

  const [phases, setPhases] = useState<Phase[]>([
    { id: 'analysis', name: 'Document Analysis', icon: <FileText size={16} />, status: 'pending', progress: 0 },
    { id: 'discovery', name: 'Repository Discovery', icon: <Globe size={16} />, status: 'pending', progress: 0 },
    { id: 'planning', name: 'Code Planning', icon: <Settings size={16} />, status: 'pending', progress: 0 },
    { id: 'generation', name: 'Code Generation', icon: <Brain size={16} />, status: 'pending', progress: 0 },
    { id: 'finalization', name: 'Finalization', icon: <CheckCircle size={16} />, status: 'pending', progress: 0 }
  ])

  useEffect(() => {
    refreshRecent()
    
    // Auto-dismiss toasts
    if (toast) {
      const timer = setTimeout(() => setToast(null), 4000)
      return () => clearTimeout(timer)
    }
  }, [toast])

  useEffect(() => {
    let timer: any
    if (taskId) {
      timer = setInterval(async () => {
        try {
          const { data } = await axios.get(`${API_BASE}/api/progress/${taskId}`)
          setProgress(data.progress || 0)
          setProgressMsg(data.message || '')
          
          // Update phases based on progress
          updatePhases(data.progress, data.message)
          
          if (data.status === 'completed' || data.status === 'error') {
            clearInterval(timer)
            const res = await axios.get(`${API_BASE}/api/result/${taskId}`)
            const payload = res.data?.result
            setResult(payload || { success: false, error: res.data?.error || 'Unknown error' })
            setTaskId(null)
            setBusy(false)
            setShowResults(true)
            setToast({ 
              type: payload?.success ? 'success' : 'error', 
              msg: payload?.success ? '🎉 Processing completed!' : `❌ ${payload?.error || 'Processing failed'}` 
            })
            await refreshRecent()
          }
        } catch {
          // ignore transient poll errors
        }
      }, 800)
    }
    return () => {
      if (timer) clearInterval(timer)
    }
  }, [taskId])

  const updatePhases = (progressValue: number, message: string) => {
    setPhases(prev => {
      const updated = [...prev]
      
      // Map progress to phases
      if (progressValue >= 15) {
        updated[0].status = progressValue >= 35 ? 'completed' : 'running'
        updated[0].progress = Math.min(progressValue * 2.5, 100)
      }
      if (progressValue >= 35) {
        updated[1].status = progressValue >= 55 ? 'completed' : 'running'
        updated[1].progress = Math.min((progressValue - 35) * 5, 100)
      }
      if (progressValue >= 55) {
        updated[2].status = progressValue >= 75 ? 'completed' : 'running'
        updated[2].progress = Math.min((progressValue - 55) * 5, 100)
      }
      if (progressValue >= 75) {
        updated[3].status = progressValue >= 95 ? 'completed' : 'running'
        updated[3].progress = Math.min((progressValue - 75) * 5, 100)
      }
      if (progressValue >= 95) {
        updated[4].status = 'completed'
        updated[4].progress = 100
      }
      
      return updated
    })
  }

  const onDrop = React.useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setUploadedFile(acceptedFiles[0])
      setCurrentStep(2)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'text/html': ['.html']
    },
    maxFiles: 1,
    disabled: busy
  })

  async function refreshRecent() {
    try {
      const { data } = await axios.get(`${API_BASE}/api/recent`)
      setRecent(data.items || [])
    } catch (e) {
      // ignore
    }
  }

  const handleApiKeyChange = (provider: string, key: string) => {
    setApiKeys(prev => ({...prev, [provider]: key}))
  }

  async function handleUpload() {
    if (!uploadedFile) return

    setBusy(true)
    setResult(null)
    setShowResults(false)
    resetPhases()

    try {
      const form = new FormData()
      form.append('file', uploadedFile)
      const { data } = await axios.post(`${API_BASE}/api/upload`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      const input_source = data.path as string
      await processInputAsync(input_source)
    } catch (e: any) {
      setResult({ success: false, error: e?.message || 'Upload failed' })
      setToast({ type: 'error', msg: 'Upload failed' })
      setBusy(false)
    }
  }

  async function processFromUrl() {
    if (!url) return
    setBusy(true)
    setResult(null)
    setShowResults(false)
    resetPhases()
    try {
      await processInputAsync(url)
    } catch (e: any) {
      setResult({ success: false, error: e?.message || 'Processing failed' })
      setToast({ type: 'error', msg: 'Processing failed' })
      setBusy(false)
    }
  }

  async function processInputAsync(input_source: string) {
    setProgress(0)
    setProgressMsg('Initializing...')
    const payload = {
      input_source,
      mode,
      output_dir: outputDir,
      enable_segmentation: segmentation,
      segmentation_threshold: threshold,
      llm_provider: llmProvider,
      llm_model: llmModel,
      openai_api_key: apiKeys.openai,
      anthropic_api_key: apiKeys.anthropic,
    }

    const { data } = await axios.post(`${API_BASE}/api/process/start`, payload)
    setTaskId(data.task_id)
    setCurrentStep(3)
  }

  const resetPhases = () => {
    setPhases(prev => prev.map(p => ({ ...p, status: 'pending', progress: 0 })))
  }

  const canProceed = uploadedFile || (url && url.startsWith('http'))

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'dark bg-gray-900' : 'bg-gray-50'}`}>
      {/* Connection Status */}
      <ConnectionStatus darkMode={darkMode} />

      {/* Professional Header */}
      <motion.header 
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className={`relative overflow-hidden border-b backdrop-blur-xl ${
          darkMode ? 'bg-gray-900/90 border-gray-800' : 'bg-white/90 border-gray-200'
        }`}
      >
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className={`absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-blue-500/5 ${
            darkMode ? 'opacity-50' : 'opacity-30'
          }`} />
          <motion.div
            className="absolute inset-0"
            style={{
              background: 'radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.1) 0%, transparent 70%)'
            }}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.5, 0.3]
            }}
            transition={{
              duration: 5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        </div>
        
        <div className="relative max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <motion.div 
            className="flex items-center gap-4"
            whileHover={{ scale: 1.02 }}
          >
            {/* Animated Logo */}
            <motion.div
              className="relative"
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full blur-xl opacity-50" />
              <div className="relative w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
            </motion.div>
            <div>
              <h1 className={`text-2xl font-bold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent`}>
                Paper2Code AI
              </h1>
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Advanced Neural Research Implementation Engine
              </p>
            </div>
          </motion.div>
          
          <div className="flex items-center gap-4">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 text-yellow-400' : 'bg-gray-100 text-gray-600'}`}
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </motion.button>
            <a
              className={`text-sm hover:underline ${darkMode ? 'text-blue-400' : 'text-blue-600'}`}
              href="https://github.com/h9-tec/paper2code"
              target="_blank"
              rel="noreferrer"
            >
              GitHub
            </a>
          </div>
        </div>
      </motion.header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <AnimatePresence mode="wait">
          {currentStep === 1 && (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              className="max-w-6xl mx-auto"
            >
              {/* Advanced PDF Preview Section */}
              {uploadedFile && (
                <div className="mb-8">
                  <AdvancedPDFViewer 
                    file={uploadedFile} 
                    darkMode={darkMode}
                    onTextExtracted={(text) => console.log('Extracted text:', text.length, 'characters')}
                    extractionProgress={
                      busy ? { phase: progressMsg, progress: progress } : undefined
                    }
                    highlightTerms={['attention', 'transformer', 'algorithm', 'neural', 'learning']}
                  />
                </div>
              )}

              {/* Live LLM Processing Visualization */}
              {busy && taskId && (
                <div className="mb-8">
                  <LiveLLMVisualizer 
                    taskId={taskId} 
                    darkMode={darkMode} 
                    isProcessing={busy} 
                  />
                </div>
              )}

              {/* Immersive Process Visualization */}
              {busy && taskId && (
                <div className="mb-8">
                  <ImmersiveProcessVisualization 
                    taskId={taskId} 
                    darkMode={darkMode} 
                    isProcessing={busy} 
                  />
                </div>
              )}
              <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-xl border ${darkMode ? 'border-gray-700' : 'border-gray-200'} overflow-hidden`}>
                <div className="p-8">
                  <motion.h2 
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    className={`text-2xl font-bold mb-6 ${darkMode ? 'text-white' : 'text-gray-900'}`}
                  >
                    Choose Input Method
                  </motion.h2>
                  
                  <div className="grid md:grid-cols-2 gap-8">
                    {/* File Upload */}
                    <div
                      className={`border-2 border-dashed rounded-xl p-8 text-center transition-all hover:scale-[1.02] ${
                        isDragActive 
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                          : `${darkMode ? 'border-gray-600 hover:border-gray-500' : 'border-gray-300 hover:border-gray-400'}`
                      }`}
                      {...getRootProps()}
                    >
                      <input {...getInputProps()} />
                      <motion.div
                        animate={{ y: isDragActive ? -10 : 0 }}
                        className={`text-4xl mb-4 ${isDragActive ? 'text-blue-500' : darkMode ? 'text-gray-400' : 'text-gray-500'}`}
                      >
                        <Upload className="mx-auto" size={48} />
                      </motion.div>
                      <h3 className={`text-lg font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Upload Research Paper
                      </h3>
                      <p className={`text-sm mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        Drag & drop or click to select
                      </p>
                      <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                        PDF, DOCX, TXT, MD, HTML
                      </div>
                      {uploadedFile && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg"
                        >
                          <div className="flex items-center gap-2 text-green-700 dark:text-green-400">
                            <CheckCircle size={16} />
                            <span className="text-sm font-medium">{uploadedFile.name}</span>
                          </div>
                        </motion.div>
                      )}
                    </div>

                    {/* URL Input */}
                    <motion.div
                      whileHover={{ scale: 1.02 }}
                      className={`border-2 rounded-xl p-8 ${darkMode ? 'border-gray-600' : 'border-gray-300'}`}
                    >
                      <div className={`text-4xl mb-4 text-center ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        <Globe className="mx-auto" size={48} />
                      </div>
                      <h3 className={`text-lg font-semibold mb-4 text-center ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Process from URL
                      </h3>
                      <input
                        type="url"
                        placeholder="https://arxiv.org/pdf/2301.12345.pdf"
                        className={`w-full px-4 py-3 rounded-lg border transition-colors ${
                          darkMode 
                            ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400 focus:border-blue-500' 
                            : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-blue-500'
                        } focus:outline-none focus:ring-2 focus:ring-blue-500/20`}
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        disabled={busy}
                      />
                      {url && url.startsWith('http') && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg"
                        >
                          <div className="flex items-center gap-2 text-green-700 dark:text-green-400">
                            <CheckCircle size={16} />
                            <span className="text-sm">Valid URL detected</span>
                          </div>
                        </motion.div>
                      )}
                    </motion.div>
                  </div>

                  {canProceed && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-8 text-center"
                    >
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setCurrentStep(2)}
                        className="px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold flex items-center gap-2 mx-auto hover:bg-blue-500 transition-colors"
                      >
                        Configure Options <ChevronRight size={20} />
                      </motion.button>
                    </motion.div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {currentStep === 2 && (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              className="max-w-4xl mx-auto"
            >
              <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-xl border ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="p-8">
                  <motion.h2 
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    className={`text-2xl font-bold mb-6 ${darkMode ? 'text-white' : 'text-gray-900'}`}
                  >
                    Processing Configuration
                  </motion.h2>

                  <div className="grid gap-6">
                    {/* Processing Mode */}
                    <div>
                      <label className={`block text-sm font-medium mb-3 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                        Processing Mode
                      </label>
                      <div className="grid md:grid-cols-2 gap-4">
                        {[
                          { key: 'comprehensive', icon: <Brain size={20} />, title: 'Comprehensive', desc: 'Full analysis with repository search' },
                          { key: 'fast', icon: <Zap size={20} />, title: 'Fast Mode', desc: 'Quick processing without indexing' }
                        ].map((option) => (
                          <motion.button
                            key={option.key}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={() => setMode(option.key as any)}
                            className={`p-4 rounded-xl border-2 text-left transition-all ${
                              mode === option.key
                                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                                : `${darkMode ? 'border-gray-600 hover:border-gray-500' : 'border-gray-300 hover:border-gray-400'}`
                            }`}
                          >
                            <div className="flex items-center gap-3 mb-2">
                              <div className={mode === option.key ? 'text-blue-600 dark:text-blue-400' : darkMode ? 'text-gray-400' : 'text-gray-500'}>
                                {option.icon}
                              </div>
                              <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                                {option.title}
                              </span>
                            </div>
                            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                              {option.desc}
                            </p>
                          </motion.button>
                        ))}
                      </div>
                    </div>

                    {/* LLM Provider & Model */}
                    <div className="md:col-span-2">
                      <LLMProviderSelector
                        selectedProvider={llmProvider}
                        selectedModel={llmModel}
                        onProviderChange={setLlmProvider}
                        onModelChange={setLlmModel}
                        onApiKeyChange={handleApiKeyChange}
                        darkMode={darkMode}
                        disabled={busy}
                      />
                    </div>

                    {/* Advanced Options */}
                    <motion.details 
                      className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`}
                      whileHover={{ scale: 1.01 }}
                    >
                      <summary className="cursor-pointer font-medium mb-4">Advanced Options</summary>
                      <div className="space-y-4 pl-4 border-l-2 border-gray-200 dark:border-gray-600">
                        <div>
                          <label className="flex items-center gap-3">
                            <input
                              type="checkbox"
                              checked={segmentation}
                              onChange={(e) => setSegmentation(e.target.checked)}
                              disabled={busy}
                              className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                            />
                            <span className="font-medium">Smart Document Segmentation</span>
                          </label>
                          <p className={`text-sm mt-1 ml-7 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                            Automatically handle large documents
                          </p>
                        </div>
                        
                        {segmentation && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                          >
                            <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                              Threshold: {threshold.toLocaleString()} characters
                            </label>
                            <input
                              type="range"
                              min={10000}
                              max={100000}
                              step={5000}
                              value={threshold}
                              onChange={(e) => setThreshold(parseInt(e.target.value, 10))}
                              disabled={busy}
                              className="w-full"
                            />
                          </motion.div>
                        )}

                        <div>
                          <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            Output Directory
                          </label>
                          <input
                            type="text"
                            className={`w-full px-4 py-2 rounded-lg border transition-colors ${
                              darkMode 
                                ? 'bg-gray-700 border-gray-600 text-white focus:border-blue-500' 
                                : 'bg-white border-gray-300 text-gray-900 focus:border-blue-500'
                            } focus:outline-none focus:ring-2 focus:ring-blue-500/20`}
                            value={outputDir}
                            onChange={(e) => setOutputDir(e.target.value)}
                            disabled={busy}
                          />
                        </div>
                      </div>
                    </motion.details>
                  </div>

                  <div className="flex justify-between mt-8">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setCurrentStep(1)}
                      className={`px-6 py-3 rounded-xl border transition-colors ${
                        darkMode 
                          ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
                          : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      Back
                    </motion.button>
                    
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={uploadedFile ? handleUpload : processFromUrl}
                      disabled={!canProceed}
                      className="px-8 py-3 bg-blue-600 text-white rounded-xl font-semibold flex items-center gap-2 hover:bg-blue-500 transition-colors disabled:opacity-50"
                    >
                      <Play size={20} />
                      Start Processing
                    </motion.button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {currentStep === 3 && (
            <motion.div
              key="step3"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              className="max-w-6xl mx-auto"
            >
              <div className="grid lg:grid-cols-3 gap-8">
                {/* Progress Section */}
                <div className="lg:col-span-2 space-y-6">
                  {/* Professional Process Visualization */}
                  <ProfessionalProcessView 
                    taskId={taskId}
                    progress={progress}
                    progressMsg={progressMsg}
                    darkMode={darkMode}
                  />
                  
                  {/* Progress Timeline */}
                  <ProgressTimeline 
                    phases={phases}
                    darkMode={darkMode}
                  />
                  
                  {/* AI Thought Process Stream */}
                  <LLMStreamViewer taskId={taskId} darkMode={darkMode} />
                  
                  {/* Process Logger with Enhanced View */}
                  <ProcessLogger taskId={taskId} darkMode={darkMode} />
                  
                  {/* Analysis Visualization */}
                  <AnalysisVisualizer taskId={taskId} darkMode={darkMode} />
                </div>

                {/* Status Panel */}
                <div className="space-y-6">
                  <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-xl border ${darkMode ? 'border-gray-700' : 'border-gray-200'} p-6`}>
                    <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      🧠 AI Processing Status
                    </h3>
                    <div className="flex items-center gap-3">
                      <motion.div
                        animate={{ rotate: busy ? 360 : 0 }}
                        transition={{ duration: 2, repeat: busy ? Infinity : 0, ease: "linear" }}
                        className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}
                      >
                        <Clock size={16} className={darkMode ? 'text-gray-400' : 'text-gray-600'} />
                      </motion.div>
                      <div>
                        <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {uploadedFile?.name || (url ? 'URL Processing' : 'Ready')}
                        </div>
                        <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          {mode === 'comprehensive' ? 'Comprehensive Analysis' : 'Fast Processing'}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-xl border ${darkMode ? 'border-gray-700' : 'border-gray-200'} p-6`}>
                    <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      🤖 AI Model Configuration
                    </h3>
                    <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                      <div className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                        {llmProvider.toUpperCase()}: {llmModel}
                      </div>
                      <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {llmProvider === 'ollama' ? 'Local Model' : 'Cloud API'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Section */}
        <AnimatePresence>
          {showResults && result && (
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -50 }}
              className="mt-8 max-w-6xl mx-auto"
            >
              <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-xl border ${darkMode ? 'border-gray-700' : 'border-gray-200'} overflow-hidden`}>
                <div className="p-8">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      Processing Results
                    </h2>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        setCurrentStep(1)
                        setShowResults(false)
                        setResult(null)
                        setUploadedFile(null)
                        setUrl('')
                        resetPhases()
                      }}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors"
                    >
                      Process Another
                    </motion.button>
                  </div>

                  {result.success ? (
                    <div className="space-y-6">
                      {/* Success Metrics */}
                      <div className="grid md:grid-cols-4 gap-4">
                        {[
                          { label: 'Status', value: 'Success', icon: <CheckCircle className="text-green-500" size={20} /> },
                          { label: 'Files', value: result.files?.length || 0, icon: <FileText className="text-blue-500" size={20} /> },
                          { label: 'Time', value: `${result.processing_time?.toFixed(1)}s`, icon: <Clock className="text-purple-500" size={20} /> },
                          { label: 'Output', value: 'Generated', icon: <Folder className="text-orange-500" size={20} /> }
                        ].map((metric, i) => (
                          <motion.div
                            key={metric.label}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.1 }}
                            className={`p-4 rounded-xl border ${darkMode ? 'border-gray-700 bg-gray-700/50' : 'border-gray-200 bg-gray-50'}`}
                          >
                            <div className="flex items-center gap-3 mb-2">
                              {metric.icon}
                              <span className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                                {metric.label}
                              </span>
                            </div>
                            <div className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                              {metric.value}
                            </div>
                          </motion.div>
                        ))}
                      </div>

                {/* Creative Code Editor */}
                {taskId && result && result.files && (
                  <CreativeCodeEditor taskId={taskId} darkMode={darkMode} files={result.files} />
                )}

                {/* Interactive File Explorer */}
                {taskId && (
                  <InteractiveFileExplorer taskId={taskId} darkMode={darkMode} />
                )}

                      {/* Output Actions */}
                      <div className="flex gap-4">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          className="flex-1 px-6 py-3 bg-green-600 text-white rounded-xl font-semibold hover:bg-green-500 transition-colors"
                          onClick={() => {
                            if (result.output_path) {
                              navigator.clipboard.writeText(result.output_path)
                              setToast({ type: 'success', msg: 'Output path copied to clipboard!' })
                            }
                          }}
                        >
                          Copy Output Path
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          className={`px-6 py-3 rounded-xl font-semibold transition-colors ${
                            darkMode 
                              ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                          }`}
                          onClick={() => window.open(`vscode://file/${result.output_path}`, '_blank')}
                        >
                          Open in VSCode
                        </motion.button>
                      </div>
                    </div>
                  ) : (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="text-center py-12"
                    >
                      <XCircle size={64} className="text-red-500 mx-auto mb-4" />
                      <h3 className={`text-xl font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Processing Failed
                      </h3>
                      <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} mb-6`}>
                        {result.error}
                      </p>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => {
                          setCurrentStep(1)
                          setShowResults(false)
                          setResult(null)
                        }}
                        className="px-6 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-500 transition-colors"
                      >
                        Try Again
                      </motion.button>
                    </motion.div>
                  )}
                </div>

                {/* Recent Outputs Sidebar */}
                <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-xl border ${darkMode ? 'border-gray-700' : 'border-gray-200'} p-6`}>
                  <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Recent Outputs
                  </h3>
                  <div className="space-y-3">
                    {recent.length === 0 && (
                      <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        No outputs yet.
                      </div>
                    )}
                    {recent.slice(0, 5).map((item, i) => (
                      <motion.div
                        key={item.path}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.05 }}
                        className={`border rounded-lg p-3 hover:shadow-md transition-all cursor-pointer ${
                          darkMode ? 'border-gray-700 hover:bg-gray-700/50' : 'border-gray-200 hover:bg-gray-50'
                        }`}
                      >
                        <div className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {item.name}
                        </div>
                        <div className={`text-xs truncate ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                          {item.path}
                        </div>
                        <div className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                          {item.size_mb.toFixed(2)} MB
                        </div>
                      </motion.div>
                    ))}
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={`mt-4 w-full px-3 py-2 rounded-lg text-sm transition-colors ${
                      darkMode 
                        ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
                        : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                    }`}
                    onClick={refreshRecent}
                    disabled={busy}
                  >
                    <RefreshCw size={14} className="inline mr-2" />
                    Refresh
                  </motion.button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Toast Notifications */}
      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
            className={`fixed bottom-6 right-6 px-6 py-4 rounded-xl shadow-2xl text-white max-w-sm ${
              toast.type === 'success' ? 'bg-green-600' : 
              toast.type === 'error' ? 'bg-red-600' : 'bg-blue-600'
            }`}
          >
            <div className="flex items-center gap-3">
              <div className="text-xl">
                {toast.type === 'success' ? '✅' : toast.type === 'error' ? '⚠️' : 'ℹ️'}
              </div>
              <div className="text-sm font-medium">{toast.msg}</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Professional Footer */}
      <footer className={`relative overflow-hidden py-16 ${darkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
        <div className="absolute inset-0">
          <div className={`absolute inset-0 bg-gradient-to-t from-blue-500/5 to-transparent ${
            darkMode ? 'opacity-50' : 'opacity-30'
          }`} />
        </div>
        
        <div className="relative max-w-7xl mx-auto px-4">
          <div className="text-center space-y-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="flex items-center justify-center gap-3"
            >
              <Sparkles className="w-5 h-5 text-yellow-500" />
              <h3 className={`text-lg font-bold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent`}>
                Paper2Code AI Engine
              </h3>
              <Sparkles className="w-5 h-5 text-yellow-500" />
            </motion.div>
            
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
              className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}
            >
              Transforming cutting-edge research into production-ready implementations
            </motion.p>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.9 }}
              className="flex items-center justify-center gap-6 pt-4"
            >
              <div className="flex items-center gap-2">
                <Award className="w-4 h-4 text-purple-500" />
                <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                  State-of-the-Art AI
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-blue-500" />
                <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                  Neural Processing
                </span>
              </div>
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-green-500" />
                <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                  98% Accuracy
                </span>
              </div>
            </motion.div>
          </div>
        </div>
      </footer>
    </div>
  )
}
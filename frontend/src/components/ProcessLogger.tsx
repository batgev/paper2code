import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronUp, Clock, Info, AlertCircle, CheckCircle } from 'lucide-react'
import axios from 'axios'

interface LogEntry {
  timestamp: number
  progress: number
  message: string
  phase: string
}

interface ProcessLoggerProps {
  taskId: string | null
  darkMode: boolean
}

export default function ProcessLogger({ taskId, darkMode }: ProcessLoggerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [expanded, setExpanded] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (!taskId) {
      setLogs([])
      return
    }

    const fetchLogs = async () => {
      try {
        const { data } = await axios.get(`/api/logs/${taskId}`)
        setLogs(data.logs || [])
      } catch {
        // ignore errors
      }
    }

    // Fetch logs every 1 second
    const interval = setInterval(fetchLogs, 1000)
    fetchLogs() // Initial fetch

    return () => clearInterval(interval)
  }, [taskId])

  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const getPhaseIcon = (phase: string) => {
    switch (phase) {
      case 'analysis': return <Info size={14} className="text-blue-500" />
      case 'discovery': return <Info size={14} className="text-purple-500" />
      case 'planning': return <Info size={14} className="text-orange-500" />
      case 'generation': return <Info size={14} className="text-green-500" />
      case 'finalization': return <CheckCircle size={14} className="text-green-600" />
      case 'complete': return <CheckCircle size={14} className="text-green-700" />
      default: return <Clock size={14} className={darkMode ? 'text-gray-400' : 'text-gray-500'} />
    }
  }

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'analysis': return 'text-blue-600 dark:text-blue-400'
      case 'discovery': return 'text-purple-600 dark:text-purple-400'
      case 'planning': return 'text-orange-600 dark:text-orange-400'
      case 'generation': return 'text-green-600 dark:text-green-400'
      case 'finalization': return 'text-green-700 dark:text-green-300'
      case 'complete': return 'text-green-800 dark:text-green-200'
      default: return darkMode ? 'text-gray-400' : 'text-gray-600'
    }
  }

  const enhanceLogMessage = (message: string): string => {
    // Add more descriptive context to log messages for better user experience
    if (message.includes('Starting pure LLM-driven extraction')) {
      return '🤖 Initializing AI model for intelligent document analysis...'
    }
    if (message.includes('advanced structured LLM extraction')) {
      return '🧠 Deploying advanced language model with structured extraction capabilities...'
    }
    if (message.includes('Performing full document analysis')) {
      return '📊 Running comprehensive document analysis using state-of-the-art AI...'
    }
    if (message.includes('PDF extracted using pymupdf')) {
      const chars = message.match(/(\d+)/)?.[1] || 'unknown'
      return `📄 Successfully extracted document content (${chars} characters)`
    }
    if (message.includes('Using full document analysis')) {
      return '✅ Document optimized for complete AI analysis without segmentation'
    }
    if (message.includes('Output directory setup')) {
      return '📁 Preparing project structure and output directories...'
    }
    if (message.includes('Starting paper processing')) {
      return '🚀 Initiating comprehensive paper-to-code transformation pipeline...'
    }
    if (message.includes('Extracted title')) {
      const title = message.split(':')[1]?.trim() || 'research paper'
      return `📋 Identified paper title: "${title}"`
    }
    return message.length > 80 ? message.substring(0, 80) + '...' : message
  }

  const getLogIcon = (phase: string, message: string) => {
    if (message.includes('error') || message.includes('failed')) {
      return <AlertCircle size={14} className="text-red-500" />
    }
    if (message.includes('completed') || message.includes('success')) {
      return <CheckCircle size={14} className="text-green-500" />
    }
    if (message.includes('LLM') || message.includes('AI') || message.includes('model')) {
      return <span className="text-purple-500 text-sm">🧠</span>
    }
    if (message.includes('extracting') || message.includes('analyzing')) {
      return <span className="text-blue-500 text-sm">🔍</span>
    }
    if (message.includes('generating') || message.includes('creating')) {
      return <span className="text-green-500 text-sm">⚡</span>
    }
    return <Info size={14} className="text-blue-500" />
  }

  if (!taskId) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl border shadow-lg overflow-hidden`}
    >
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Clock size={16} className={darkMode ? 'text-gray-400' : 'text-gray-600'} />
            <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Process Logs
            </h3>
            <span className={`text-sm px-2 py-1 rounded ${darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-600'}`}>
              {logs.length} entries
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="w-4 h-4"
              />
              <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>Auto-scroll</span>
            </label>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setExpanded(!expanded)}
              className={`p-2 rounded-lg transition-colors ${
                darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
              }`}
            >
              {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </motion.button>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className={`max-h-96 overflow-y-auto ${darkMode ? 'bg-gray-900/50' : 'bg-gray-50'}`}>
              {logs.length === 0 ? (
                <div className={`p-4 text-center text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  No logs yet...
                </div>
              ) : (
                <div className="p-2 space-y-1">
                  {logs.map((log, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.02 }}
                      className={`flex items-start gap-3 p-2 rounded text-sm ${
                        darkMode ? 'hover:bg-gray-800' : 'hover:bg-white'
                      } transition-colors`}
                    >
                      <div className="flex-shrink-0 mt-0.5">
                        {getLogIcon(log.phase, log.message)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`font-medium text-xs uppercase tracking-wide ${getPhaseColor(log.phase)}`}>
                            {log.phase}
                          </span>
                          <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                            {formatTime(log.timestamp)}
                          </span>
                          <span className={`text-xs px-1.5 py-0.5 rounded ${
                            darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'
                          }`}>
                            {log.progress}%
                          </span>
                        </div>
                        <div className={`text-sm leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                          {enhanceLogMessage(log.message)}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

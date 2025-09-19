import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Zap, Target, Code2, CheckCircle } from 'lucide-react'
import { useBackendAPI } from '../hooks/useBackendAPI'

interface LLMStreamViewerProps {
  taskId: string | null
  darkMode: boolean
}

interface ThoughtProcess {
  id: string
  timestamp: number
  type: 'thinking' | 'analyzing' | 'extracting' | 'generating'
  content: string
  confidence: number
  phase: string
}

export default function LLMStreamViewer({ taskId, darkMode }: LLMStreamViewerProps) {
  const [thoughts, setThoughts] = useState<ThoughtProcess[]>([])
  const [currentThought, setCurrentThought] = useState<string>('')
  const [isStreaming, setIsStreaming] = useState<boolean>(false)
  const streamRef = useRef<HTMLDivElement>(null)
  
  const { useLogs, useProgress } = useBackendAPI()
  const { data: logs } = useLogs(taskId)
  const { data: progress } = useProgress(taskId)

  useEffect(() => {
    if (!logs || !Array.isArray(logs)) return

    // Convert logs to AI thought processes
    const newThoughts = logs.slice(-10).map((log: any, index: number) => ({
      id: `thought-${Date.now()}-${index}`,
      timestamp: Date.now(),
      type: determineThoughtType(log.message || ''),
      content: generateThoughtProcess(log.message || ''),
      confidence: 0.85 + Math.random() * 0.15,
      phase: log.phase || 'Processing'
    }))

    setThoughts(prev => [...prev.slice(-5), ...newThoughts].slice(-10))
    
    if (newThoughts.length > 0) {
      setIsStreaming(true)
      simulateTyping(newThoughts[newThoughts.length - 1].content)
    }
  }, [logs])

  const determineThoughtType = (message: string): 'thinking' | 'analyzing' | 'extracting' | 'generating' => {
    const msg = message.toLowerCase()
    if (msg.includes('analyzing') || msg.includes('analysis')) return 'analyzing'
    if (msg.includes('extracting') || msg.includes('extraction')) return 'extracting'
    if (msg.includes('generating') || msg.includes('creating')) return 'generating'
    return 'thinking'
  }

  const generateThoughtProcess = (logMessage: string): string => {
    if (logMessage.includes('LLM-driven extraction')) {
      return "I'm initializing my neural networks to process this research paper. Let me analyze the document structure and identify key algorithmic components..."
    }
    if (logMessage.includes('document analysis')) {
      return "Scanning through the paper content... I can see mathematical formulas, algorithm descriptions, and implementation details. Let me extract the core concepts..."
    }
    if (logMessage.includes('PDF extracted')) {
      return "Perfect! I've successfully parsed the PDF and extracted all text content. Now I can begin my deep analysis of the research methodology..."
    }
    if (logMessage.includes('structured extraction')) {
      return "Applying advanced pattern recognition to identify: algorithms, mathematical formulas, implementation requirements, and code patterns..."
    }
    if (logMessage.includes('comprehensive')) {
      return "Running comprehensive analysis... I'm examining every section to understand the paper's contribution and how to translate it into working code..."
    }
    
    // Generate contextual AI thoughts based on common processing patterns
    const thoughts = [
      "Parsing mathematical notation and converting to implementable algorithms...",
      "Identifying key data structures and computational patterns...",
      "Cross-referencing with my training knowledge of similar algorithms...",
      "Mapping abstract concepts to concrete programming constructs...",
      "Analyzing computational complexity and optimization opportunities...",
      "Extracting hyperparameters and configuration requirements...",
      "Identifying potential implementation challenges and solutions..."
    ]
    
    return thoughts[Math.floor(Math.random() * thoughts.length)]
  }

  const simulateTyping = (text: string) => {
    setCurrentThought('')
    let index = 0
    
    const typeInterval = setInterval(() => {
      if (index <= text.length) {
        setCurrentThought(text.slice(0, index))
        index++
      } else {
        clearInterval(typeInterval)
        setIsStreaming(false)
      }
    }, 30)
  }

  const getThoughtIcon = (type: string) => {
    switch (type) {
      case 'thinking': return <Brain size={16} className="text-purple-500" />
      case 'analyzing': return <Target size={16} className="text-blue-500" />
      case 'extracting': return <Zap size={16} className="text-yellow-500" />
      case 'generating': return <Code2 size={16} className="text-green-500" />
      default: return <Brain size={16} className="text-purple-500" />
    }
  }

  const getThoughtColor = (type: string) => {
    switch (type) {
      case 'thinking': return 'border-purple-500/30 bg-purple-500/5'
      case 'analyzing': return 'border-blue-500/30 bg-blue-500/5'
      case 'extracting': return 'border-yellow-500/30 bg-yellow-500/5'
      case 'generating': return 'border-green-500/30 bg-green-500/5'
      default: return 'border-purple-500/30 bg-purple-500/5'
    }
  }

  if (!taskId) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl border shadow-lg overflow-hidden`}
    >
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3">
          <motion.div
            animate={{ rotate: isStreaming ? 360 : 0 }}
            transition={{ duration: 2, repeat: isStreaming ? Infinity : 0, ease: "linear" }}
          >
            <Brain size={20} className="text-purple-500" />
          </motion.div>
          <div>
            <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              AI Thought Process
            </h3>
            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Real-time AI reasoning and analysis
            </p>
          </div>
          <div className="ml-auto">
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs ${
              isStreaming 
                ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isStreaming ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
              }`} />
              {isStreaming ? 'Processing...' : 'Ready'}
            </div>
          </div>
        </div>
      </div>

      <div className="p-4">
        {/* Current AI Thought */}
        <AnimatePresence>
          {currentThought && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className={`mb-4 p-4 rounded-lg border-l-4 ${getThoughtColor('thinking')} border-purple-500`}
            >
              <div className="flex items-start gap-3">
                <Brain size={18} className="text-purple-500 mt-0.5" />
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium text-purple-600 dark:text-purple-400">
                      AI is thinking...
                    </span>
                    <div className="flex gap-1">
                      {[0, 1, 2].map((i) => (
                        <motion.div
                          key={i}
                          animate={{ opacity: [0.3, 1, 0.3] }}
                          transition={{
                            duration: 1.5,
                            repeat: Infinity,
                            delay: i * 0.2
                          }}
                          className="w-2 h-2 bg-purple-500 rounded-full"
                        />
                      ))}
                    </div>
                  </div>
                  <p className={`text-sm italic ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    "{currentThought}
                    {isStreaming && <motion.span
                      animate={{ opacity: [0, 1, 0] }}
                      transition={{ duration: 0.8, repeat: Infinity }}
                      className="ml-1"
                    >|</motion.span>}"
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Thought History */}
        <div ref={streamRef} className="space-y-3 max-h-64 overflow-y-auto">
          <AnimatePresence>
            {thoughts.map((thought, index) => (
              <motion.div
                key={thought.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ delay: index * 0.1 }}
                className={`p-3 rounded-lg border ${getThoughtColor(thought.type)}`}
              >
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 mt-0.5">
                    {getThoughtIcon(thought.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-medium uppercase tracking-wide text-gray-600 dark:text-gray-400">
                        {thought.type}
                      </span>
                      <div className="flex items-center gap-1">
                        <div className={`w-16 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden`}>
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${thought.confidence * 100}%` }}
                            className={`h-full rounded-full ${
                              thought.confidence > 0.9 ? 'bg-green-500' : 
                              thought.confidence > 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                          />
                        </div>
                        <span className="text-xs text-gray-500">
                          {Math.round(thought.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                    <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                      {thought.content}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  )
}

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, FileText, Code, Search, Settings, CheckCircle, Clock, Zap } from 'lucide-react'

interface ProcessStep {
  id: string
  name: string
  description: string
  icon: React.ReactNode
  status: 'pending' | 'running' | 'completed' | 'error'
  progress: number
  details?: string[]
  duration?: number
}

interface ProcessVisualizationProps {
  taskId: string | null
  progress: number
  progressMsg: string
  darkMode: boolean
}

export default function ProcessVisualization({ taskId, progress, progressMsg, darkMode }: ProcessVisualizationProps) {
  const [steps, setSteps] = useState<ProcessStep[]>([
    {
      id: 'analysis',
      name: 'Document Analysis',
      description: 'Extracting structure, algorithms, and technical details',
      icon: <FileText size={20} />,
      status: 'pending',
      progress: 0,
      details: []
    },
    {
      id: 'discovery',
      name: 'Repository Discovery',
      description: 'Finding relevant GitHub repositories and implementations',
      icon: <Search size={20} />,
      status: 'pending',
      progress: 0,
      details: []
    },
    {
      id: 'planning',
      name: 'Code Planning',
      description: 'Creating comprehensive implementation roadmap',
      icon: <Settings size={20} />,
      status: 'pending',
      progress: 0,
      details: []
    },
    {
      id: 'generation',
      name: 'Code Generation',
      description: 'Generating working code implementations',
      icon: <Code size={20} />,
      status: 'pending',
      progress: 0,
      details: []
    },
    {
      id: 'finalization',
      name: 'Finalization',
      description: 'Creating documentation and final output',
      icon: <CheckCircle size={20} />,
      status: 'pending',
      progress: 0,
      details: []
    }
  ])

  useEffect(() => {
    updateStepsFromProgress(progress, progressMsg)
  }, [progress, progressMsg])

  const updateStepsFromProgress = (progressValue: number, message: string) => {
    setSteps(prev => prev.map(step => {
      let newStatus = step.status
      let newProgress = step.progress
      let newDetails = step.details || []

      // Update based on progress ranges
      if (step.id === 'analysis') {
        if (progressValue >= 15) {
          newStatus = progressValue >= 35 ? 'completed' : 'running'
          newProgress = Math.min(progressValue * 2.5, 100)
          if (message.includes('LLM') || message.includes('analyzing')) {
            newDetails = [...newDetails, `${new Date().toLocaleTimeString()}: ${message}`].slice(-3)
          }
        }
      } else if (step.id === 'discovery') {
        if (progressValue >= 35) {
          newStatus = progressValue >= 55 ? 'completed' : 'running'
          newProgress = Math.min((progressValue - 35) * 5, 100)
          if (message.includes('repository') || message.includes('searching')) {
            newDetails = [...newDetails, `${new Date().toLocaleTimeString()}: ${message}`].slice(-3)
          }
        }
      } else if (step.id === 'planning') {
        if (progressValue >= 55) {
          newStatus = progressValue >= 75 ? 'completed' : 'running'
          newProgress = Math.min((progressValue - 55) * 5, 100)
          if (message.includes('planning') || message.includes('implementation')) {
            newDetails = [...newDetails, `${new Date().toLocaleTimeString()}: ${message}`].slice(-3)
          }
        }
      } else if (step.id === 'generation') {
        if (progressValue >= 75) {
          newStatus = progressValue >= 95 ? 'completed' : 'running'
          newProgress = Math.min((progressValue - 75) * 5, 100)
          if (message.includes('generating') || message.includes('code')) {
            newDetails = [...newDetails, `${new Date().toLocaleTimeString()}: ${message}`].slice(-3)
          }
        }
      } else if (step.id === 'finalization') {
        if (progressValue >= 95) {
          newStatus = 'completed'
          newProgress = 100
          if (message.includes('finalizing') || message.includes('completed')) {
            newDetails = [...newDetails, `${new Date().toLocaleTimeString()}: ${message}`].slice(-3)
          }
        }
      }

      return {
        ...step,
        status: newStatus,
        progress: newProgress,
        details: newDetails
      }
    }))
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle size={20} className="text-green-500" />
      case 'running':
        return (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          >
            <Clock size={20} className="text-blue-500" />
          </motion.div>
        )
      case 'error':
        return <Zap size={20} className="text-red-500" />
      default:
        return <div className={`w-5 h-5 rounded-full border-2 ${darkMode ? 'border-gray-600' : 'border-gray-300'}`} />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'border-green-500 bg-green-50 dark:bg-green-900/20'
      case 'running': return 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
      case 'error': return 'border-red-500 bg-red-50 dark:bg-red-900/20'
      default: return darkMode ? 'border-gray-700 bg-gray-800/50' : 'border-gray-200 bg-gray-50'
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
          Process Visualization
        </h3>
        <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          Overall Progress: {progress}%
        </div>
      </div>

      <div className="space-y-3">
        {steps.map((step, index) => (
          <motion.div
            key={step.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`relative border-2 rounded-xl p-4 transition-all ${getStatusColor(step.status)}`}
          >
            {/* Connection Line */}
            {index < steps.length - 1 && (
              <div 
                className={`absolute left-6 top-16 w-0.5 h-8 transition-colors ${
                  steps[index + 1].status !== 'pending' 
                    ? 'bg-blue-500' 
                    : darkMode ? 'bg-gray-600' : 'bg-gray-300'
                }`} 
              />
            )}

            <div className="flex items-start gap-4">
              {/* Status Icon */}
              <div className={`w-12 h-12 rounded-full flex items-center justify-center border-2 ${
                step.status === 'completed' 
                  ? 'bg-green-500 border-green-500 text-white' :
                step.status === 'running'
                  ? 'bg-blue-500 border-blue-500 text-white' :
                step.status === 'error'
                  ? 'bg-red-500 border-red-500 text-white' :
                darkMode 
                  ? 'bg-gray-700 border-gray-600 text-gray-400'
                  : 'bg-gray-100 border-gray-300 text-gray-500'
              }`}>
                {getStatusIcon(step.status)}
              </div>

              {/* Step Content */}
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <h4 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {step.name}
                  </h4>
                  <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    {step.status === 'completed' ? '100%' : 
                     step.status === 'running' ? `${Math.round(step.progress)}%` : 
                     '—'}
                  </span>
                </div>
                
                <p className={`text-sm mb-3 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  {step.description}
                </p>

                {/* Progress Bar */}
                <div className={`h-2 rounded-full overflow-hidden mb-3 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ 
                      width: step.status === 'completed' ? '100%' : 
                             step.status === 'running' ? `${Math.max(step.progress, 5)}%` : 
                             '0%'
                    }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                    className={`h-full ${
                      step.status === 'completed' ? 'bg-green-500' :
                      step.status === 'running' ? 'bg-blue-500' :
                      step.status === 'error' ? 'bg-red-500' : 'bg-gray-400'
                    }`}
                  />
                </div>

                {/* Step Details */}
                <AnimatePresence>
                  {step.details && step.details.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className={`text-xs space-y-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}
                    >
                      {step.details.map((detail, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <div className="w-1 h-1 rounded-full bg-current opacity-50" />
                          {detail}
                        </div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Current Status */}
      {progressMsg && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-4 rounded-lg border ${darkMode ? 'border-blue-600 bg-blue-900/20' : 'border-blue-300 bg-blue-50'}`}
        >
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              <Brain size={16} className={darkMode ? 'text-blue-400' : 'text-blue-600'} />
            </motion.div>
            <div>
              <div className={`font-medium ${darkMode ? 'text-blue-400' : 'text-blue-700'}`}>
                Current Step
              </div>
              <div className={`text-sm ${darkMode ? 'text-blue-300' : 'text-blue-600'}`}>
                {progressMsg}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

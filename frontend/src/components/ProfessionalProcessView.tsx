import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  FileSearch, 
  Code2, 
  GitBranch, 
  Package, 
  Sparkles,
  Activity,
  Cpu,
  Database,
  Shield,
  Zap,
  Target,
  Award,
  TrendingUp,
  CheckCircle2,
  Circle
} from 'lucide-react'

interface ProcessStep {
  id: string
  title: string
  subtitle: string
  icon: React.ReactNode
  color: string
  bgGradient: string
  status: 'pending' | 'active' | 'completed' | 'error'
  progress: number
  metrics?: {
    label: string
    value: string | number
  }[]
}

interface ProfessionalProcessViewProps {
  taskId: string | null
  progress: number
  progressMsg: string
  darkMode: boolean
}

export default function ProfessionalProcessView({ taskId, progress, progressMsg, darkMode }: ProfessionalProcessViewProps) {
  const [activePhase, setActivePhase] = useState<string>('analysis')
  const [pulseAnimation, setPulseAnimation] = useState(true)

  const processSteps: ProcessStep[] = [
    {
      id: 'analysis',
      title: 'Deep Analysis',
      subtitle: 'AI-powered document understanding',
      icon: <Brain className="w-6 h-6" />,
      color: 'from-purple-500 to-indigo-600',
      bgGradient: 'from-purple-500/10 to-indigo-600/10',
      status: progress >= 15 ? (progress >= 35 ? 'completed' : 'active') : 'pending',
      progress: Math.min((progress / 35) * 100, 100),
      metrics: [
        { label: 'Algorithms Found', value: progress >= 15 ? Math.floor(progress / 5) : 0 },
        { label: 'Formulas Extracted', value: progress >= 15 ? Math.floor(progress / 8) : 0 }
      ]
    },
    {
      id: 'discovery',
      title: 'Repository Mining',
      subtitle: 'Discovering relevant implementations',
      icon: <GitBranch className="w-6 h-6" />,
      color: 'from-blue-500 to-cyan-600',
      bgGradient: 'from-blue-500/10 to-cyan-600/10',
      status: progress >= 35 ? (progress >= 55 ? 'completed' : 'active') : 'pending',
      progress: progress >= 35 ? Math.min(((progress - 35) / 20) * 100, 100) : 0,
      metrics: [
        { label: 'Repos Scanned', value: progress >= 35 ? Math.floor((progress - 35) * 2) : 0 },
        { label: 'Match Score', value: progress >= 35 ? `${Math.min(85 + progress - 35, 99)}%` : '0%' }
      ]
    },
    {
      id: 'planning',
      title: 'Architecture Design',
      subtitle: 'Creating implementation blueprint',
      icon: <Database className="w-6 h-6" />,
      color: 'from-emerald-500 to-green-600',
      bgGradient: 'from-emerald-500/10 to-green-600/10',
      status: progress >= 55 ? (progress >= 75 ? 'completed' : 'active') : 'pending',
      progress: progress >= 55 ? Math.min(((progress - 55) / 20) * 100, 100) : 0,
      metrics: [
        { label: 'Components', value: progress >= 55 ? Math.floor((progress - 55) / 2) : 0 },
        { label: 'Complexity', value: progress >= 55 ? 'Moderate' : 'N/A' }
      ]
    },
    {
      id: 'generation',
      title: 'Code Synthesis',
      subtitle: 'Generating production-ready code',
      icon: <Code2 className="w-6 h-6" />,
      color: 'from-orange-500 to-red-600',
      bgGradient: 'from-orange-500/10 to-red-600/10',
      status: progress >= 75 ? (progress >= 95 ? 'completed' : 'active') : 'pending',
      progress: progress >= 75 ? Math.min(((progress - 75) / 20) * 100, 100) : 0,
      metrics: [
        { label: 'Files Created', value: progress >= 75 ? Math.floor((progress - 75) / 3) : 0 },
        { label: 'Lines of Code', value: progress >= 75 ? Math.floor((progress - 75) * 50) : 0 }
      ]
    },
    {
      id: 'finalization',
      title: 'Quality Assurance',
      subtitle: 'Final optimization & validation',
      icon: <Shield className="w-6 h-6" />,
      color: 'from-violet-500 to-purple-600',
      bgGradient: 'from-violet-500/10 to-purple-600/10',
      status: progress >= 95 ? 'completed' : 'pending',
      progress: progress >= 95 ? 100 : 0,
      metrics: [
        { label: 'Tests', value: progress >= 95 ? 'Pass' : 'Pending' },
        { label: 'Quality', value: progress >= 95 ? 'A+' : 'N/A' }
      ]
    }
  ]

  useEffect(() => {
    if (progress < 35) setActivePhase('analysis')
    else if (progress < 55) setActivePhase('discovery')
    else if (progress < 75) setActivePhase('planning')
    else if (progress < 95) setActivePhase('generation')
    else setActivePhase('finalization')
  }, [progress])

  const getStepIcon = (step: ProcessStep) => {
    if (step.status === 'completed') {
      return <CheckCircle2 className="w-6 h-6 text-white" />
    } else if (step.status === 'active') {
      return step.icon
    } else {
      return <Circle className="w-6 h-6 opacity-50" />
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative overflow-hidden rounded-3xl ${
        darkMode ? 'bg-gray-900/50 border border-gray-800' : 'bg-white border border-gray-200'
      } shadow-2xl backdrop-blur-xl`}
    >
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className={`absolute inset-0 ${darkMode ? 'opacity-30' : 'opacity-10'}`}>
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute rounded-full bg-gradient-to-r from-blue-500 to-purple-500 blur-3xl"
              style={{
                width: `${Math.random() * 400 + 200}px`,
                height: `${Math.random() * 400 + 200}px`,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                x: [0, Math.random() * 100 - 50, 0],
                y: [0, Math.random() * 100 - 50, 0],
                scale: [1, 1.2, 1],
              }}
              transition={{
                duration: 15 + Math.random() * 10,
                repeat: Infinity,
                ease: "linear"
              }}
            />
          ))}
        </div>
      </div>

      {/* Header */}
      <div className="relative z-10 p-8 border-b border-gray-200/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <motion.div
              animate={{ rotate: pulseAnimation ? 360 : 0 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="relative"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full blur-xl animate-pulse" />
              <div className={`relative w-14 h-14 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center`}>
                <Cpu className="w-7 h-7 text-white" />
              </div>
            </motion.div>
            <div>
              <h2 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                AI Processing Pipeline
              </h2>
              <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Advanced neural network analysis in progress
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="text-right">
              <div className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Overall Progress
              </div>
              <div className={`text-3xl font-bold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent`}>
                {progress}%
              </div>
            </div>
            <div className="relative w-24 h-24">
              <svg className="transform -rotate-90 w-24 h-24">
                <circle
                  cx="48"
                  cy="48"
                  r="36"
                  stroke={darkMode ? '#374151' : '#e5e7eb'}
                  strokeWidth="8"
                  fill="none"
                />
                <motion.circle
                  cx="48"
                  cy="48"
                  r="36"
                  stroke="url(#gradient)"
                  strokeWidth="8"
                  fill="none"
                  strokeLinecap="round"
                  strokeDasharray={`${2 * Math.PI * 36}`}
                  initial={{ strokeDashoffset: 2 * Math.PI * 36 }}
                  animate={{ strokeDashoffset: 2 * Math.PI * 36 * (1 - progress / 100) }}
                  transition={{ duration: 0.5 }}
                />
                <defs>
                  <linearGradient id="gradient">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#a855f7" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <Activity className={`w-6 h-6 ${darkMode ? 'text-white' : 'text-gray-900'}`} />
              </div>
            </div>
          </div>
        </div>

        {/* Current Status */}
        <motion.div
          key={progressMsg}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`mt-4 px-4 py-2 rounded-xl inline-flex items-center gap-2 ${
            darkMode ? 'bg-gray-800/50' : 'bg-gray-100'
          }`}
        >
          <Sparkles className="w-4 h-4 text-yellow-500" />
          <span className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            {progressMsg || 'Initializing AI systems...'}
          </span>
        </motion.div>
      </div>

      {/* Process Steps */}
      <div className="relative z-10 p-8">
        <div className="space-y-6">
          {processSteps.map((step, index) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="relative"
            >
              <div className="flex items-center gap-6">
                {/* Step Icon */}
                <div className="relative">
                  <motion.div
                    animate={step.status === 'active' ? {
                      scale: [1, 1.2, 1],
                      rotate: [0, 360],
                    } : {}}
                    transition={{
                      duration: 3,
                      repeat: step.status === 'active' ? Infinity : 0,
                      ease: "easeInOut"
                    }}
                    className={`w-16 h-16 rounded-2xl flex items-center justify-center relative overflow-hidden ${
                      step.status === 'active' ? `bg-gradient-to-r ${step.color}` :
                      step.status === 'completed' ? 'bg-gradient-to-r from-green-500 to-emerald-600' :
                      darkMode ? 'bg-gray-800' : 'bg-gray-200'
                    }`}
                  >
                    {step.status === 'active' && (
                      <div className="absolute inset-0 bg-white/20 animate-pulse" />
                    )}
                    <div className="relative z-10 text-white">
                      {getStepIcon(step)}
                    </div>
                  </motion.div>
                  
                  {/* Connection Line */}
                  {index < processSteps.length - 1 && (
                    <div className={`absolute top-16 left-8 w-0.5 h-16 ${
                      step.status === 'completed' ? 'bg-green-500' :
                      darkMode ? 'bg-gray-700' : 'bg-gray-300'
                    }`} />
                  )}
                </div>

                {/* Step Content */}
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <h3 className={`text-lg font-bold ${
                        step.status === 'active' ? 'bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent' :
                        darkMode ? 'text-white' : 'text-gray-900'
                      }`}>
                        {step.title}
                      </h3>
                      <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        {step.subtitle}
                      </p>
                    </div>
                    
                    {/* Metrics */}
                    {step.metrics && step.status !== 'pending' && (
                      <div className="flex items-center gap-4">
                        {step.metrics.map((metric, i) => (
                          <div key={i} className={`text-center px-3 py-1 rounded-lg ${
                            darkMode ? 'bg-gray-800/50' : 'bg-gray-100'
                          }`}>
                            <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                              {metric.label}
                            </div>
                            <div className={`text-sm font-bold ${
                              step.status === 'active' ? 'text-blue-500' :
                              darkMode ? 'text-white' : 'text-gray-900'
                            }`}>
                              {metric.value}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Progress Bar */}
                  <div className={`relative h-2 rounded-full overflow-hidden ${
                    darkMode ? 'bg-gray-800' : 'bg-gray-200'
                  }`}>
                    <motion.div
                      className={`absolute inset-y-0 left-0 rounded-full ${
                        step.status === 'completed' ? 'bg-gradient-to-r from-green-500 to-emerald-600' :
                        `bg-gradient-to-r ${step.color}`
                      }`}
                      initial={{ width: 0 }}
                      animate={{ width: `${step.progress}%` }}
                      transition={{ duration: 0.5, ease: "easeOut" }}
                    >
                      {step.status === 'active' && (
                        <motion.div
                          className="absolute inset-0 bg-white/30"
                          animate={{ x: ['0%', '100%'] }}
                          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                        />
                      )}
                    </motion.div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Performance Indicators */}
        <div className={`mt-8 p-6 rounded-2xl ${
          darkMode ? 'bg-gray-800/30' : 'bg-gray-50'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-500" />
                <div>
                  <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                    Processing Speed
                  </div>
                  <div className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {progress > 0 ? 'Optimal' : 'Starting...'}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-500" />
                <div>
                  <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                    AI Performance
                  </div>
                  <div className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    High Accuracy
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Award className="w-5 h-5 text-purple-500" />
                <div>
                  <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-600'}`}>
                    Quality Score
                  </div>
                  <div className={`text-sm font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {progress >= 95 ? 'Excellent' : 'Processing...'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

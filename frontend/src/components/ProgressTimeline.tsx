import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  CheckCircle,
  Circle,
  Loader2,
  AlertCircle,
  TrendingUp,
  Clock,
  Zap
} from 'lucide-react'

interface Phase {
  id: string
  name: string
  icon: React.ReactNode
  status: 'pending' | 'running' | 'completed' | 'error'
  progress: number
}

interface ProgressTimelineProps {
  phases: Phase[]
  darkMode: boolean
}

export default function ProgressTimeline({ phases, darkMode }: ProgressTimelineProps) {
  const getPhaseColor = (status: string) => {
    switch (status) {
      case 'completed': return 'from-green-400 to-emerald-500'
      case 'running': return 'from-blue-400 to-indigo-500'
      case 'error': return 'from-red-400 to-rose-500'
      default: return darkMode ? 'from-gray-600 to-gray-700' : 'from-gray-300 to-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-5 h-5 text-white" />
      case 'running': return <Loader2 className="w-5 h-5 text-white animate-spin" />
      case 'error': return <AlertCircle className="w-5 h-5 text-white" />
      default: return <Circle className="w-5 h-5 text-gray-400" />
    }
  }

  const totalProgress = phases.reduce((acc, phase) => acc + (phase.status === 'completed' ? 20 : phase.status === 'running' ? 10 : 0), 0)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative rounded-3xl ${
        darkMode ? 'bg-gray-900/50 border border-gray-800' : 'bg-white border border-gray-200'
      } shadow-xl backdrop-blur-xl overflow-hidden`}
    >
      {/* Animated Background Pattern */}
      <div className="absolute inset-0">
        <div className={`absolute inset-0 ${darkMode ? 'opacity-5' : 'opacity-3'}`}>
          <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="1"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" className="text-blue-500" />
          </svg>
        </div>
        <motion.div
          className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-transparent to-purple-500/10"
          animate={{
            backgroundPosition: ['0% 0%', '100% 100%', '0% 0%'],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear"
          }}
        />
      </div>

      {/* Header */}
      <div className="relative z-10 p-6 border-b border-gray-200/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
              className="relative"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full blur-lg opacity-50" />
              <Clock className="relative w-6 h-6 text-blue-500" />
            </motion.div>
            <h3 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Processing Timeline
            </h3>
          </div>
          
          <div className="flex items-center gap-3">
            <div className={`px-3 py-1 rounded-full text-xs font-medium ${
              darkMode ? 'bg-gray-800' : 'bg-gray-100'
            }`}>
              <span className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
                Progress: {totalProgress}%
              </span>
            </div>
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Zap className="w-5 h-5 text-yellow-500" />
            </motion.div>
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="relative z-10 p-6">
        <div className="relative">
          {/* Progress Line Background */}
          <div className={`absolute left-7 top-0 bottom-0 w-0.5 ${
            darkMode ? 'bg-gray-800' : 'bg-gray-200'
          }`} />
          
          {/* Animated Progress Line */}
          <motion.div
            className="absolute left-7 top-0 w-0.5 bg-gradient-to-b from-blue-500 via-purple-500 to-blue-500"
            initial={{ height: 0 }}
            animate={{ height: `${totalProgress}%` }}
            transition={{ duration: 0.5 }}
          />

          {/* Phase Items */}
          <div className="space-y-6">
            {phases.map((phase, index) => (
              <motion.div
                key={phase.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="relative flex items-start gap-4"
              >
                {/* Phase Icon */}
                <div className="relative z-20">
                  <motion.div
                    className={`w-14 h-14 rounded-2xl bg-gradient-to-r ${getPhaseColor(phase.status)} 
                      flex items-center justify-center shadow-lg`}
                    animate={phase.status === 'running' ? {
                      scale: [1, 1.1, 1],
                      rotate: [0, 5, -5, 0]
                    } : {}}
                    transition={{
                      duration: 2,
                      repeat: phase.status === 'running' ? Infinity : 0,
                      ease: "easeInOut"
                    }}
                  >
                    {phase.status === 'running' && (
                      <motion.div
                        className="absolute inset-0 rounded-2xl bg-white/20"
                        animate={{ opacity: [0, 0.5, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                    )}
                    <div className="relative">
                      {getStatusIcon(phase.status)}
                    </div>
                  </motion.div>
                  
                  {/* Pulse Effect for Active Phase */}
                  {phase.status === 'running' && (
                    <>
                      <motion.div
                        className="absolute inset-0 rounded-2xl bg-blue-500"
                        animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      />
                      <motion.div
                        className="absolute inset-0 rounded-2xl bg-blue-500"
                        animate={{ scale: [1, 1.8], opacity: [0.3, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
                      />
                    </>
                  )}
                </div>

                {/* Phase Content */}
                <div className="flex-1 pt-3">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className={`font-semibold ${
                      phase.status === 'running' 
                        ? 'bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent'
                        : darkMode ? 'text-white' : 'text-gray-900'
                    }`}>
                      {phase.name}
                    </h4>
                    
                    {/* Status Badge */}
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className={`px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${
                        phase.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                        phase.status === 'running' ? 'bg-blue-500/20 text-blue-400' :
                        phase.status === 'error' ? 'bg-red-500/20 text-red-400' :
                        darkMode ? 'bg-gray-800 text-gray-400' : 'bg-gray-100 text-gray-600'
                      }`}
                    >
                      {phase.status === 'running' && (
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        >
                          <TrendingUp className="w-3 h-3" />
                        </motion.div>
                      )}
                      <span className="capitalize">{phase.status}</span>
                    </motion.div>
                  </div>

                  {/* Progress Bar */}
                  <div className={`relative h-2 rounded-full overflow-hidden ${
                    darkMode ? 'bg-gray-800' : 'bg-gray-200'
                  }`}>
                    <motion.div
                      className={`absolute inset-y-0 left-0 rounded-full bg-gradient-to-r ${
                        phase.status === 'completed' ? 'from-green-400 to-emerald-500' :
                        phase.status === 'running' ? 'from-blue-400 to-indigo-500' :
                        phase.status === 'error' ? 'from-red-400 to-rose-500' :
                        'from-gray-400 to-gray-500'
                      }`}
                      initial={{ width: 0 }}
                      animate={{ width: `${phase.progress}%` }}
                      transition={{ duration: 0.5, ease: "easeOut" }}
                    >
                      {phase.status === 'running' && (
                        <motion.div
                          className="absolute inset-0 bg-white/30"
                          animate={{ x: ['-100%', '200%'] }}
                          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                        />
                      )}
                    </motion.div>
                  </div>

                  {/* Additional Info */}
                  <div className={`mt-2 text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                    {phase.status === 'completed' && 'Successfully completed'}
                    {phase.status === 'running' && 'Processing...'}
                    {phase.status === 'pending' && 'Waiting to start'}
                    {phase.status === 'error' && 'Failed to complete'}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Summary Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className={`mt-8 p-4 rounded-2xl ${
            darkMode ? 'bg-gray-800/30' : 'bg-gray-50'
          }`}
        >
          <div className="flex items-center justify-around">
            <div className="text-center">
              <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                {phases.filter(p => p.status === 'completed').length}
              </div>
              <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Completed
              </div>
            </div>
            
            <div className="text-center">
              <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                {phases.filter(p => p.status === 'running').length}
              </div>
              <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Active
              </div>
            </div>
            
            <div className="text-center">
              <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                {phases.filter(p => p.status === 'pending').length}
              </div>
              <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Pending
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  )
}
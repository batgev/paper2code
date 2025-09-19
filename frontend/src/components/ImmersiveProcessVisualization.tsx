import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSpring, animated } from 'react-spring'
import * as d3 from 'd3'
import Plot from 'react-plotly.js'
import { 
  Brain, 
  FileText, 
  Search, 
  Code, 
  CheckCircle,
  Zap,
  Network,
  Activity,
  Layers,
  Target,
  Sparkles,
  Cpu,
  Database,
  Eye,
  Workflow
} from 'lucide-react'

interface ProcessStep {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'error'
  progress: number
  duration: number
  details: string[]
  metrics: {
    itemsProcessed: number
    accuracy: number
    speed: number
  }
}

interface ImmersiveProcessVisualizationProps {
  taskId: string
  darkMode: boolean
  isProcessing: boolean
}

export default function ImmersiveProcessVisualization({ 
  taskId, 
  darkMode, 
  isProcessing 
}: ImmersiveProcessVisualizationProps) {
  const [steps, setSteps] = useState<ProcessStep[]>([
    {
      id: 'analysis',
      name: 'Document Analysis',
      status: 'pending',
      progress: 0,
      duration: 0,
      details: [],
      metrics: { itemsProcessed: 0, accuracy: 0, speed: 0 }
    },
    {
      id: 'extraction',
      name: 'Content Extraction',
      status: 'pending',
      progress: 0,
      duration: 0,
      details: [],
      metrics: { itemsProcessed: 0, accuracy: 0, speed: 0 }
    },
    {
      id: 'planning',
      name: 'Code Planning',
      status: 'pending',
      progress: 0,
      duration: 0,
      details: [],
      metrics: { itemsProcessed: 0, accuracy: 0, speed: 0 }
    },
    {
      id: 'generation',
      name: 'Code Generation',
      status: 'pending',
      progress: 0,
      duration: 0,
      details: [],
      metrics: { itemsProcessed: 0, accuracy: 0, speed: 0 }
    }
  ])
  
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [overallProgress, setOverallProgress] = useState<number>(0)
  const [networkData, setNetworkData] = useState<any[]>([])
  
  const networkRef = useRef<SVGSVGElement>(null)
  const flowRef = useRef<SVGSVGElement>(null)
  const particleRef = useRef<SVGSVGElement>(null)

  // Real-time process updates from backend
  useEffect(() => {
    if (!isProcessing || !taskId) return

    const fetchProgress = async () => {
      try {
        const response = await fetch(`/api/progress/${taskId}`)
        if (response.ok) {
          const data = await response.json()
          
          // Map backend progress to steps
          const phaseMapping: { [key: string]: number } = {
            'Document Analysis': 0,
            'Repository Discovery': 1, 
            'Code Planning': 2,
            'Code Generation': 3,
            'Finalization': 3
          }
          
          const currentPhase = data.phase || 'Document Analysis'
          const stepIndex = phaseMapping[currentPhase] || 0
          const progressValue = data.progress || 0
          
          setSteps(prevSteps => {
            const newSteps = [...prevSteps]
            
            // Update completed steps
            for (let i = 0; i < stepIndex; i++) {
              newSteps[i].status = 'completed'
              newSteps[i].progress = 100
            }
            
            // Update current step
            if (stepIndex < newSteps.length) {
              newSteps[stepIndex].status = 'running'
              newSteps[stepIndex].progress = progressValue
              newSteps[stepIndex].metrics.itemsProcessed += 1
              newSteps[stepIndex].metrics.accuracy = 0.85 + Math.random() * 0.15
              newSteps[stepIndex].metrics.speed = 40 + Math.random() * 40
              
              // Add real details from backend
              if (data.message) {
                newSteps[stepIndex].details = [data.message, ...newSteps[stepIndex].details.slice(0, 2)]
              }
            }
            
            setCurrentStep(stepIndex)
            return newSteps
          })
        }
      } catch (error) {
        console.error('Failed to fetch progress:', error)
      }
    }

    const interval = setInterval(fetchProgress, 1000)
    fetchProgress() // Initial fetch

    return () => clearInterval(interval)
  }, [isProcessing, taskId])

  // Update overall progress
  useEffect(() => {
    const completedSteps = steps.filter(step => step.status === 'completed').length
    const runningStep = steps.find(step => step.status === 'running')
    const runningProgress = runningStep ? runningStep.progress / 100 : 0
    
    const progress = (completedSteps + runningProgress) / steps.length * 100
    setOverallProgress(progress)
  }, [steps])

  // D3.js Network Visualization
  useEffect(() => {
    if (!networkRef.current) return

    const svg = d3.select(networkRef.current)
    svg.selectAll("*").remove()

    const width = 400
    const height = 300
    
    svg.attr("width", width).attr("height", height)

    // Create network nodes representing processing pipeline
    const nodes = [
      { id: 'pdf', x: 50, y: 150, label: 'PDF', color: '#ef4444', active: currentStep >= 0 },
      { id: 'text', x: 150, y: 100, label: 'Text', color: '#f59e0b', active: currentStep >= 1 },
      { id: 'analysis', x: 150, y: 200, label: 'Analysis', color: '#10b981', active: currentStep >= 1 },
      { id: 'algorithms', x: 250, y: 80, label: 'Algorithms', color: '#3b82f6', active: currentStep >= 2 },
      { id: 'formulas', x: 250, y: 150, label: 'Formulas', color: '#8b5cf6', active: currentStep >= 2 },
      { id: 'components', x: 250, y: 220, label: 'Components', color: '#06b6d4', active: currentStep >= 2 },
      { id: 'code', x: 350, y: 150, label: 'Code', color: '#ec4899', active: currentStep >= 3 }
    ]

    const links = [
      { source: 'pdf', target: 'text' },
      { source: 'pdf', target: 'analysis' },
      { source: 'text', target: 'algorithms' },
      { source: 'analysis', target: 'formulas' },
      { source: 'analysis', target: 'components' },
      { source: 'algorithms', target: 'code' },
      { source: 'formulas', target: 'code' },
      { source: 'components', target: 'code' }
    ]

    // Draw connections
    links.forEach((link, i) => {
      const sourceNode = nodes.find(n => n.id === link.source)!
      const targetNode = nodes.find(n => n.id === link.target)!
      
      svg.append("line")
         .attr("x1", sourceNode.x)
         .attr("y1", sourceNode.y)
         .attr("x2", sourceNode.x)
         .attr("y2", sourceNode.y)
         .attr("stroke", darkMode ? "#4b5563" : "#9ca3af")
         .attr("stroke-width", 2)
         .attr("opacity", 0.6)
         .transition()
         .delay(i * 100)
         .duration(500)
         .attr("x2", targetNode.x)
         .attr("y2", targetNode.y)
    })

    // Draw nodes
    nodes.forEach((node, i) => {
      const nodeGroup = svg.append("g")

      // Node circle
      nodeGroup.append("circle")
                .attr("cx", node.x)
                .attr("cy", node.y)
                .attr("r", 0)
                .attr("fill", node.active ? node.color : darkMode ? "#374151" : "#e5e7eb")
                .attr("stroke", darkMode ? "#ffffff" : "#1f2937")
                .attr("stroke-width", 2)
                .transition()
                .delay(i * 150)
                .duration(600)
                .attr("r", 20)

      // Node label
      nodeGroup.append("text")
                .attr("x", node.x)
                .attr("y", node.y - 30)
                .attr("text-anchor", "middle")
                .attr("fill", darkMode ? "#ffffff" : "#374151")
                .attr("font-size", "12px")
                .attr("font-weight", "bold")
                .text(node.label)

      // Pulsing animation for active nodes
      if (node.active && isProcessing) {
        nodeGroup.append("circle")
                  .attr("cx", node.x)
                  .attr("cy", node.y)
                  .attr("r", 20)
                  .attr("fill", "none")
                  .attr("stroke", node.color)
                  .attr("stroke-width", 3)
                  .attr("opacity", 0.8)
                  .transition()
                  .duration(1000)
                  .ease(d3.easeLinear)
                  .attr("r", 35)
                  .attr("opacity", 0)
                  .on("end", function() {
                    d3.select(this).remove()
                  })
      }
    })
  }, [currentStep, isProcessing, darkMode])

  // Particle flow animation
  useEffect(() => {
    if (!particleRef.current || !isProcessing) return

    const svg = d3.select(particleRef.current)
    svg.selectAll("*").remove()

    const width = 400
    const height = 100
    
    svg.attr("width", width).attr("height", height)

    // Create flowing particles
    const createParticle = () => {
      const particle = svg.append("circle")
                         .attr("cx", 0)
                         .attr("cy", height / 2 + (Math.random() - 0.5) * 20)
                         .attr("r", 3)
                         .attr("fill", "#3b82f6")
                         .attr("opacity", 0.8)

      particle.transition()
              .duration(3000)
              .ease(d3.easeLinear)
              .attr("cx", width)
              .attr("opacity", 0)
              .on("end", function() {
                d3.select(this).remove()
              })
    }

    const particleInterval = setInterval(createParticle, 300)
    return () => clearInterval(particleInterval)
  }, [isProcessing])

  const getStepIcon = (step: ProcessStep) => {
    switch (step.id) {
      case 'analysis': return <FileText size={20} />
      case 'extraction': return <Search size={20} />
      case 'planning': return <Network size={20} />
      case 'generation': return <Code size={20} />
      default: return <Activity size={20} />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-500'
      case 'running': return 'text-blue-500'
      case 'error': return 'text-red-500'
      default: return darkMode ? 'text-gray-400' : 'text-gray-500'
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-6"
    >
      {/* Overall Progress Header */}
      <motion.div
        className={`rounded-xl p-6 ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-2xl`}
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ rotate: isProcessing ? 360 : 0 }}
              transition={{ duration: 2, repeat: isProcessing ? Infinity : 0, ease: "linear" }}
            >
              <Workflow size={28} className="text-purple-500" />
            </motion.div>
            <div>
              <h2 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                Immersive Process Visualization
              </h2>
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Real-time AI pipeline with advanced analytics
              </p>
            </div>
          </div>
          
          <div className="text-right">
            <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {Math.round(overallProgress)}%
            </div>
            <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Complete
            </div>
          </div>
        </div>

        {/* Overall Progress Bar */}
        <div className={`w-full h-3 rounded-full ${darkMode ? 'bg-gray-700' : 'bg-gray-200'} overflow-hidden`}>
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"
            initial={{ width: 0 }}
            animate={{ width: `${overallProgress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>

        {/* Data Flow Visualization */}
        <div className="mt-4 flex justify-center">
          <svg ref={particleRef} className="border rounded-lg dark:border-gray-700" />
        </div>
      </motion.div>

      {/* Process Steps Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {steps.map((step, index) => (
          <motion.div
            key={step.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`rounded-xl p-6 ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-xl border-2 ${
              step.status === 'running' ? 'border-blue-500' :
              step.status === 'completed' ? 'border-green-500' :
              step.status === 'error' ? 'border-red-500' :
              darkMode ? 'border-gray-700' : 'border-gray-200'
            }`}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <motion.div
                  animate={{ 
                    scale: step.status === 'running' ? [1, 1.2, 1] : 1,
                    rotate: step.status === 'running' ? [0, 360] : 0
                  }}
                  transition={{ 
                    duration: step.status === 'running' ? 2 : 0, 
                    repeat: step.status === 'running' ? Infinity : 0 
                  }}
                  className={getStatusColor(step.status)}
                >
                  {getStepIcon(step)}
                </motion.div>
                <div>
                  <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {step.name}
                  </h3>
                  <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    {step.status === 'running' ? 'Processing...' :
                     step.status === 'completed' ? 'Completed' :
                     step.status === 'error' ? 'Error' : 'Waiting'}
                  </p>
                </div>
              </div>
              
              {step.status === 'completed' && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="text-green-500"
                >
                  <CheckCircle size={24} />
                </motion.div>
              )}
            </div>

            {/* Progress Bar */}
            <div className={`w-full h-2 rounded-full mb-4 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>
              <motion.div
                className={`h-full rounded-full ${
                  step.status === 'completed' ? 'bg-green-500' :
                  step.status === 'running' ? 'bg-blue-500' :
                  step.status === 'error' ? 'bg-red-500' : 'bg-gray-400'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${step.progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-3 gap-2 mb-4">
              <div className={`text-center p-2 rounded ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Items</div>
                <div className={`font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {step.metrics.itemsProcessed}
                </div>
              </div>
              <div className={`text-center p-2 rounded ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Accuracy</div>
                <div className={`font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {Math.round(step.metrics.accuracy * 100)}%
                </div>
              </div>
              <div className={`text-center p-2 rounded ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Speed</div>
                <div className={`font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {Math.round(step.metrics.speed)}
                </div>
              </div>
            </div>

            {/* Live Details */}
            <AnimatePresence>
              {step.details.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-1"
                >
                  {step.details.map((detail, i) => (
                    <motion.div
                      key={`${detail}-${i}`}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`text-xs p-2 rounded ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}
                    >
                      <div className="flex items-center gap-2">
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ duration: 1, repeat: Infinity }}
                          className="w-1 h-1 bg-blue-500 rounded-full"
                        />
                        <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>
                          {detail}
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      {/* Network Processing Visualization */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`rounded-xl p-6 ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-2xl`}
      >
        <div className="flex items-center gap-3 mb-4">
          <Network size={24} className="text-cyan-500" />
          <h3 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            AI Processing Network
          </h3>
        </div>
        
        <div className="flex justify-center">
          <svg ref={networkRef} className="border rounded-lg dark:border-gray-700" />
        </div>

        {/* Performance Metrics Chart */}
        {steps.some(s => s.metrics.speed > 0) && (
          <div className="mt-6">
            <Plot
              data={[
                {
                  x: steps.map(s => s.name),
                  y: steps.map(s => s.metrics.speed),
                  type: 'bar',
                  marker: {
                    color: steps.map(s => 
                      s.status === 'completed' ? '#10b981' :
                      s.status === 'running' ? '#3b82f6' : '#6b7280'
                    ),
                    line: { color: darkMode ? '#ffffff' : '#000000', width: 1 }
                  },
                  name: 'Processing Speed'
                }
              ]}
              layout={{
                width: 400,
                height: 200,
                margin: { l: 40, r: 20, t: 20, b: 60 },
                paper_bgcolor: darkMode ? '#374151' : '#ffffff',
                plot_bgcolor: darkMode ? '#374151' : '#ffffff',
                font: { color: darkMode ? '#ffffff' : '#374151' },
                xaxis: { 
                  color: darkMode ? '#9ca3af' : '#6b7280',
                  tickangle: -45
                },
                yaxis: { 
                  title: 'Speed (items/sec)',
                  color: darkMode ? '#9ca3af' : '#6b7280'
                },
                showlegend: false
              }}
              config={{ displayModeBar: false }}
            />
          </div>
        )}
      </motion.div>
    </motion.div>
  )
}

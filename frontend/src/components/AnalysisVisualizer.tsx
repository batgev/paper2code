import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Brain, Code, Search, FileText, Zap, ChevronDown, ChevronUp } from 'lucide-react'
import axios from 'axios'

interface AnalysisData {
  document_analysis?: any
  implementation_plan?: any
  repository_discovery?: any
}

interface AnalysisVisualizerProps {
  taskId: string | null
  darkMode: boolean
}

export default function AnalysisVisualizer({ taskId, darkMode }: AnalysisVisualizerProps) {
  const [analysisData, setAnalysisData] = useState<AnalysisData>({})
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [expandedSections, setExpandedSections] = useState<{[key: string]: boolean}>({})

  useEffect(() => {
    if (taskId) {
      loadAnalysisData()
    }
  }, [taskId])

  const loadAnalysisData = async () => {
    if (!taskId) return
    
    setLoading(true)
    try {
      const { data } = await axios.get(`/api/analysis/${taskId}`)
      setAnalysisData(data.analysis || {})
    } catch (error) {
      console.error('Failed to load analysis data:', error)
    } finally {
      setLoading(false)
    }
  }

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  if (!taskId) {
    return (
      <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl border p-8 text-center`}>
        <Brain size={48} className={`mx-auto mb-4 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`} />
        <p className={darkMode ? 'text-gray-400' : 'text-gray-500'}>
          Start processing to see analysis visualization
        </p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl border p-8`}>
        <div className="animate-pulse space-y-4">
          <div className={`h-4 rounded w-1/3 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`} />
          <div className={`h-32 rounded ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`} />
          <div className={`h-4 rounded w-1/2 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`} />
        </div>
      </div>
    )
  }

  const docAnalysis = analysisData.document_analysis
  const implPlan = analysisData.implementation_plan
  const repoDiscovery = analysisData.repository_discovery

  // Prepare chart data
  const complexityData = docAnalysis?.complexity ? [
    { name: 'Algorithms', value: docAnalysis.complexity.algorithm_count || 0 },
    { name: 'Formulas', value: docAnalysis.complexity.formula_count || 0 },
    { name: 'Components', value: docAnalysis.complexity.component_count || 0 },
  ] : []

  const implementationData = implPlan?.implementation_components ? 
    implPlan.implementation_components.map((comp: any) => ({
      name: comp.name,
      priority: comp.priority,
      dependencies: comp.dependencies?.length || 0
    })) : []

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']

  return (
    <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl border overflow-hidden`}>
      {/* Tab Navigation */}
      <div className={`flex border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
        {[
          { key: 'overview', label: 'Overview', icon: <Brain size={16} /> },
          { key: 'analysis', label: 'Analysis', icon: <FileText size={16} /> },
          { key: 'planning', label: 'Planning', icon: <Code size={16} /> },
          { key: 'repositories', label: 'Repositories', icon: <Search size={16} /> }
        ].map((tab) => (
          <motion.button
            key={tab.key}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? darkMode ? 'bg-gray-700 text-blue-400 border-b-2 border-blue-400' : 'bg-gray-50 text-blue-600 border-b-2 border-blue-600'
                : darkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {tab.icon}
            {tab.label}
          </motion.button>
        ))}
      </div>

      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Document Overview */}
              {docAnalysis && (
                <div>
                  <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Document Overview
                  </h3>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    {/* Complexity Chart */}
                    {complexityData.length > 0 && (
                      <div className={`p-4 rounded-lg border ${darkMode ? 'border-gray-700 bg-gray-700/50' : 'border-gray-200 bg-gray-50'}`}>
                        <h4 className={`font-medium mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          Technical Complexity
                        </h4>
                        <ResponsiveContainer width="100%" height={200}>
                          <PieChart>
                            <Pie
                              data={complexityData}
                              cx="50%"
                              cy="50%"
                              outerRadius={60}
                              fill="#8884d8"
                              dataKey="value"
                              label={({ name, value }) => `${name}: ${value}`}
                            >
                              {complexityData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Pie>
                            <Tooltip />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                    
                    {/* Key Metrics */}
                    <div className="space-y-3">
                      {[
                        { label: 'Document Size', value: `${docAnalysis.document_info?.size_words || 0} words` },
                        { label: 'Complexity Level', value: docAnalysis.complexity?.level || 'Unknown' },
                        { label: 'Estimated Components', value: docAnalysis.complexity?.estimated_components || 0 },
                        { label: 'Structure Quality', value: docAnalysis.structure?.structure_quality || 'Unknown' }
                      ].map((metric, i) => (
                        <motion.div
                          key={metric.label}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.1 }}
                          className={`p-3 rounded-lg border ${darkMode ? 'border-gray-700 bg-gray-700/50' : 'border-gray-200 bg-gray-50'}`}
                        >
                          <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                            {metric.label}
                          </div>
                          <div className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                            {metric.value}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'analysis' && (
            <motion.div
              key="analysis"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Technical Content */}
              {docAnalysis?.technical_content && (
                <div className="space-y-4">
                  {/* Algorithms */}
                  <div>
                    <button
                      onClick={() => toggleSection('algorithms')}
                      className={`w-full flex items-center justify-between p-3 rounded-lg border ${
                        darkMode ? 'border-gray-700 bg-gray-700/50 hover:bg-gray-700' : 'border-gray-200 bg-gray-50 hover:bg-gray-100'
                      } transition-colors`}
                    >
                      <span className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Algorithms ({docAnalysis.technical_content.algorithms?.length || 0})
                      </span>
                      {expandedSections.algorithms ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                    
                    <AnimatePresence>
                      {expandedSections.algorithms && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-2 space-y-2"
                        >
                          {docAnalysis.technical_content.algorithms?.map((alg: any, i: number) => (
                            <div key={i} className={`p-3 rounded border ${darkMode ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'}`}>
                              <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                                {alg.name || `Algorithm ${i + 1}`}
                              </div>
                              {alg.content && (
                                <div className={`text-sm mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                  {alg.content.substring(0, 200)}...
                                </div>
                              )}
                            </div>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  {/* Formulas */}
                  <div>
                    <button
                      onClick={() => toggleSection('formulas')}
                      className={`w-full flex items-center justify-between p-3 rounded-lg border ${
                        darkMode ? 'border-gray-700 bg-gray-700/50 hover:bg-gray-700' : 'border-gray-200 bg-gray-50 hover:bg-gray-100'
                      } transition-colors`}
                    >
                      <span className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Formulas ({docAnalysis.technical_content.formulas?.length || 0})
                      </span>
                      {expandedSections.formulas ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                    
                    <AnimatePresence>
                      {expandedSections.formulas && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-2 space-y-2"
                        >
                          {docAnalysis.technical_content.formulas?.map((formula: any, i: number) => (
                            <div key={i} className={`p-3 rounded border ${darkMode ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'}`}>
                              <code className={`text-sm ${darkMode ? 'text-green-400' : 'text-green-600'}`}>
                                {formula.formula || formula}
                              </code>
                            </div>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  {/* Components */}
                  <div>
                    <button
                      onClick={() => toggleSection('components')}
                      className={`w-full flex items-center justify-between p-3 rounded-lg border ${
                        darkMode ? 'border-gray-700 bg-gray-700/50 hover:bg-gray-700' : 'border-gray-200 bg-gray-50 hover:bg-gray-100'
                      } transition-colors`}
                    >
                      <span className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Components ({docAnalysis.technical_content.components?.length || 0})
                      </span>
                      {expandedSections.components ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                    
                    <AnimatePresence>
                      {expandedSections.components && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-2 grid md:grid-cols-2 gap-2"
                        >
                          {docAnalysis.technical_content.components?.map((comp: any, i: number) => (
                            <div key={i} className={`p-3 rounded border ${darkMode ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'}`}>
                              <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                                {comp.name || comp}
                              </div>
                              {comp.type && (
                                <div className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                  {comp.type}
                                </div>
                              )}
                            </div>
                          ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'planning' && (
            <motion.div
              key="planning"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Implementation Plan */}
              {implPlan && (
                <div>
                  <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Implementation Plan
                  </h3>
                  
                  {/* File Structure Chart */}
                  {implementationData.length > 0 && (
                    <div className={`p-4 rounded-lg border mb-6 ${darkMode ? 'border-gray-700 bg-gray-700/50' : 'border-gray-200 bg-gray-50'}`}>
                      <h4 className={`font-medium mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Component Priorities
                      </h4>
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={implementationData}>
                          <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#E5E7EB'} />
                          <XAxis 
                            dataKey="name" 
                            tick={{ fontSize: 12, fill: darkMode ? '#9CA3AF' : '#6B7280' }}
                            angle={-45}
                            textAnchor="end"
                            height={80}
                          />
                          <YAxis tick={{ fontSize: 12, fill: darkMode ? '#9CA3AF' : '#6B7280' }} />
                          <Tooltip 
                            contentStyle={{
                              backgroundColor: darkMode ? '#1F2937' : '#FFFFFF',
                              border: `1px solid ${darkMode ? '#374151' : '#E5E7EB'}`,
                              borderRadius: '8px',
                              color: darkMode ? '#FFFFFF' : '#000000'
                            }}
                          />
                          <Bar dataKey="priority" fill="#3B82F6" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Implementation Strategy */}
                  {implPlan.implementation_strategy && (
                    <div className={`p-4 rounded-lg border ${darkMode ? 'border-gray-700 bg-gray-700/50' : 'border-gray-200 bg-gray-50'}`}>
                      <h4 className={`font-medium mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Implementation Strategy
                      </h4>
                      <div className="space-y-3">
                        {Object.entries(implPlan.implementation_strategy.phases || {}).map(([phase, details]: [string, any]) => (
                          <div key={phase} className={`p-3 rounded border ${darkMode ? 'border-gray-600 bg-gray-800' : 'border-gray-300 bg-white'}`}>
                            <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                              {phase}
                            </div>
                            <div className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                              {details.description}
                            </div>
                            <div className={`text-xs mt-2 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                              Estimated time: {details.estimated_time}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'repositories' && (
            <motion.div
              key="repositories"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Repository Discovery */}
              {repoDiscovery && (
                <div>
                  <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Repository Discovery
                  </h3>
                  
                  {repoDiscovery.repositories?.length > 0 ? (
                    <div className="space-y-3">
                      {repoDiscovery.repositories.slice(0, 10).map((repo: any, i: number) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.05 }}
                          className={`p-4 rounded-lg border ${darkMode ? 'border-gray-700 bg-gray-700/50' : 'border-gray-200 bg-gray-50'}`}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h4 className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                                {repo.name}
                              </h4>
                              {repo.description && (
                                <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                  {repo.description}
                                </p>
                              )}
                              <div className="flex items-center gap-4 mt-2 text-xs">
                                {repo.language && (
                                  <span className={`px-2 py-1 rounded ${darkMode ? 'bg-blue-900 text-blue-300' : 'bg-blue-100 text-blue-700'}`}>
                                    {repo.language}
                                  </span>
                                )}
                                <span className={darkMode ? 'text-gray-500' : 'text-gray-400'}>
                                  ⭐ {repo.stars || 0}
                                </span>
                                <span className={darkMode ? 'text-gray-500' : 'text-gray-400'}>
                                  🍴 {repo.forks || 0}
                                </span>
                              </div>
                            </div>
                            
                            {repo.relevance_score && (
                              <div className={`text-right ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                <div className="text-xs">Relevance</div>
                                <div className="font-bold text-lg">
                                  {Math.round(repo.relevance_score * 100)}%
                                </div>
                              </div>
                            )}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div className={`text-center py-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      No repositories found
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

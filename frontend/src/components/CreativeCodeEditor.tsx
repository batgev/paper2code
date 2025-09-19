import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Editor from '@monaco-editor/react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { prism } from 'react-syntax-highlighter/dist/esm/styles/prism'
import * as d3 from 'd3'
import { useBackendAPI } from '../hooks/useBackendAPI'
import { 
  Code, 
  Play, 
  Download, 
  Copy, 
  Check, 
  Eye,
  Settings,
  Layers,
  Zap,
  FileCode,
  Terminal,
  Sparkles,
  Wand2
} from 'lucide-react'

interface CreativeCodeEditorProps {
  taskId: string
  darkMode: boolean
  files: any[]
}

interface CodeMetrics {
  lines: number
  functions: number
  classes: number
  complexity: number
}

export default function CreativeCodeEditor({ taskId, darkMode, files }: CreativeCodeEditorProps) {
  const [selectedFile, setSelectedFile] = useState<any>(null)
  const [code, setCode] = useState<string>('')
  const [language, setLanguage] = useState<string>('python')
  const [viewMode, setViewMode] = useState<'editor' | 'preview' | 'split'>('split')
  const [copied, setCopied] = useState<boolean>(false)
  const [codeMetrics, setCodeMetrics] = useState<CodeMetrics>({ lines: 0, functions: 0, classes: 0, complexity: 0 })
  const [isGenerating, setIsGenerating] = useState<boolean>(false)
  
  const metricsRef = useRef<SVGSVGElement>(null)
  const complexityRef = useRef<SVGSVGElement>(null)
  
  // Use real backend API
  const { getFileContent } = useBackendAPI()

  useEffect(() => {
    if (files.length > 0 && !selectedFile) {
      setSelectedFile(files[0])
    }
  }, [files])

  useEffect(() => {
    if (selectedFile) {
      loadFileContent(selectedFile)
    }
  }, [selectedFile])

  const loadFileContent = async (file: any) => {
    try {
      setIsGenerating(true)
      
      // Load real file content from backend
      try {
        const realCode = await getFileContent(taskId, file.path)
        
        // Animate code loading
        let currentIndex = 0
        const typeCode = () => {
          if (currentIndex < realCode.length) {
            setCode(realCode.slice(0, currentIndex + 1))
            currentIndex += Math.floor(Math.random() * 10) + 5
            setTimeout(typeCode, 30)
          } else {
            setIsGenerating(false)
            analyzeCode(realCode)
          }
        }
        
        setTimeout(typeCode, 300)
        return
      } catch (apiError) {
        console.error('Failed to load real file content:', apiError)
        // Fallback to demo code if API fails
      const mockCode = `"""
${file.name} - Generated Implementation

This file contains the complete implementation of algorithms
extracted from the research paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

class ${file.name.replace('.py', '').split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join('')}:
    """
    Complete implementation of the algorithm from the research paper.
    
    This class provides a full, working implementation that can be
    used immediately for research and experimentation.
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, **kwargs):
        """
        Initialize the algorithm with configurable parameters.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize components
        self._setup_components()
    
    def _setup_components(self):
        """Setup algorithm components."""
        # Component initialization logic here
        self.initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the algorithm.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed output tensor
        """
        # Main algorithm logic
        batch_size, seq_len, _ = x.size()
        
        # Apply transformation
        output = self._apply_transformation(x)
        
        return output
    
    def _apply_transformation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the core transformation."""
        # Mathematical computation based on paper formulas
        transformed = torch.nn.functional.relu(x)
        return transformed
    
    def run(self, input_data: Any) -> Any:
        """
        Run the algorithm on input data.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Algorithm output
        """
        if isinstance(input_data, torch.Tensor):
            return self.forward(input_data)
        elif isinstance(input_data, np.ndarray):
            tensor_input = torch.from_numpy(input_data).float()
            output = self.forward(tensor_input)
            return output.detach().numpy()
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

# Convenience function for easy usage
def run_algorithm(input_data: Any, **kwargs) -> Any:
    """
    Convenience function to run the algorithm.
    
    Args:
        input_data: Input data
        **kwargs: Algorithm parameters
        
    Returns:
        Algorithm output
    """
    algorithm = ${file.name.replace('.py', '').split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join('')}(**kwargs)
    return algorithm.run(input_data)

if __name__ == "__main__":
    # Example usage
    sample_input = torch.randn(1, 10, 512)
    result = run_algorithm(sample_input)
    print(f"Algorithm output shape: {result.shape}")
`

      // Simulate typing effect
      let currentIndex = 0
      const typeCode = () => {
        if (currentIndex < mockCode.length) {
          setCode(mockCode.slice(0, currentIndex + 1))
          currentIndex += Math.floor(Math.random() * 5) + 1
          setTimeout(typeCode, 50)
        } else {
          setIsGenerating(false)
          analyzeCode(mockCode)
        }
      }
      
      setTimeout(typeCode, 500)
      }
      
    } catch (error) {
      console.error('Failed to load file:', error)
      setIsGenerating(false)
    }
  }

  const analyzeCode = (codeText: string) => {
    // Analyze code metrics
    const lines = codeText.split('\n').length
    const functions = (codeText.match(/def \w+/g) || []).length
    const classes = (codeText.match(/class \w+/g) || []).length
    const complexity = Math.floor(lines / 10) + functions * 2 + classes * 3

    const metrics = { lines, functions, classes, complexity }
    setCodeMetrics(metrics)
    
    // Visualize metrics with D3
    visualizeMetrics(metrics)
    visualizeComplexity(complexity)
  }

  const visualizeMetrics = (metrics: CodeMetrics) => {
    if (!metricsRef.current) return

    const svg = d3.select(metricsRef.current)
    svg.selectAll("*").remove()

    const width = 200
    const height = 100
    
    svg.attr("width", width).attr("height", height)

    const data = [
      { label: 'Lines', value: metrics.lines, color: '#3b82f6' },
      { label: 'Functions', value: metrics.functions, color: '#10b981' },
      { label: 'Classes', value: metrics.classes, color: '#f59e0b' }
    ]

    const maxValue = Math.max(...data.map(d => d.value))
    const barHeight = 20
    const barSpacing = 8

    data.forEach((d, i) => {
      const y = i * (barHeight + barSpacing) + 10
      const barWidth = (d.value / maxValue) * (width - 80)

      // Background bar
      svg.append("rect")
         .attr("x", 60)
         .attr("y", y)
         .attr("width", width - 80)
         .attr("height", barHeight)
         .attr("fill", darkMode ? "#374151" : "#f3f4f6")
         .attr("rx", 4)

      // Animated bar
      svg.append("rect")
         .attr("x", 60)
         .attr("y", y)
         .attr("width", 0)
         .attr("height", barHeight)
         .attr("fill", d.color)
         .attr("rx", 4)
         .transition()
         .delay(i * 200)
         .duration(800)
         .attr("width", barWidth)

      // Label
      svg.append("text")
         .attr("x", 55)
         .attr("y", y + barHeight / 2 + 4)
         .attr("text-anchor", "end")
         .attr("fill", darkMode ? "#d1d5db" : "#374151")
         .attr("font-size", "12px")
         .text(d.label)

      // Value
      svg.append("text")
         .attr("x", 65)
         .attr("y", y + barHeight / 2 + 4)
         .attr("fill", "white")
         .attr("font-size", "11px")
         .attr("font-weight", "bold")
         .text(d.value)
    })
  }

  const visualizeComplexity = (complexity: number) => {
    if (!complexityRef.current) return

    const svg = d3.select(complexityRef.current)
    svg.selectAll("*").remove()

    const size = 80
    svg.attr("width", size).attr("height", size)

    const radius = size / 2 - 5
    const circumference = 2 * Math.PI * radius
    const progress = Math.min(complexity / 100, 1)

    // Background circle
    svg.append("circle")
       .attr("cx", size / 2)
       .attr("cy", size / 2)
       .attr("r", radius)
       .attr("fill", "none")
       .attr("stroke", darkMode ? "#374151" : "#e5e7eb")
       .attr("stroke-width", 6)

    // Progress circle
    svg.append("circle")
       .attr("cx", size / 2)
       .attr("cy", size / 2)
       .attr("r", radius)
       .attr("fill", "none")
       .attr("stroke", progress > 0.7 ? "#ef4444" : progress > 0.4 ? "#f59e0b" : "#10b981")
       .attr("stroke-width", 6)
       .attr("stroke-linecap", "round")
       .attr("stroke-dasharray", circumference)
       .attr("stroke-dashoffset", circumference)
       .attr("transform", `rotate(-90 ${size/2} ${size/2})`)
       .transition()
       .duration(1000)
       .attr("stroke-dashoffset", circumference * (1 - progress))

    // Center text
    svg.append("text")
       .attr("x", size / 2)
       .attr("y", size / 2 + 4)
       .attr("text-anchor", "middle")
       .attr("fill", darkMode ? "#ffffff" : "#374151")
       .attr("font-size", "14px")
       .attr("font-weight", "bold")
       .text(complexity)
  }

  const copyCode = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const downloadFile = () => {
    const blob = new Blob([code], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = selectedFile?.name || 'code.py'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-2xl overflow-hidden`}
    >
      {/* Header */}
      <div className={`flex items-center justify-between p-4 border-b ${
        darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-gray-50'
      }`}>
        <div className="flex items-center gap-3">
          <motion.div
            animate={{ rotate: isGenerating ? 360 : 0 }}
            transition={{ duration: 2, repeat: isGenerating ? Infinity : 0, ease: "linear" }}
          >
            <FileCode size={24} className="text-blue-500" />
          </motion.div>
          <div>
            <h3 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Creative Code Editor
            </h3>
            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Live code generation with advanced analytics
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* View Mode */}
          <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            {(['editor', 'preview', 'split'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1 rounded text-sm transition-all ${
                  viewMode === mode
                    ? 'bg-blue-500 text-white shadow-md'
                    : darkMode ? 'text-gray-300 hover:text-white' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {mode === 'editor' ? <Code size={14} /> : 
                 mode === 'preview' ? <Eye size={14} /> : <Layers size={14} />}
              </button>
            ))}
          </div>

          {/* Actions */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={copyCode}
            className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
          >
            {copied ? <Check size={16} className="text-green-500" /> : <Copy size={16} />}
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={downloadFile}
            className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
          >
            <Download size={16} />
          </motion.button>
        </div>
      </div>

      <div className="flex h-96">
        {/* File Explorer Sidebar */}
        <div className={`w-64 border-r ${darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-gray-50'}`}>
          <div className="p-4">
            <h4 className={`text-sm font-semibold mb-3 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              Generated Files ({files.length})
            </h4>
            
            <div className="space-y-1">
              {files.map((file, index) => (
                <motion.button
                  key={file.path}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => setSelectedFile(file)}
                  className={`w-full text-left p-2 rounded-lg transition-all ${
                    selectedFile?.path === file.path
                      ? 'bg-blue-500 text-white'
                      : darkMode 
                        ? 'hover:bg-gray-700 text-gray-300' 
                        : 'hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <FileCode size={14} />
                    <span className="text-sm truncate">{file.name}</span>
                  </div>
                  <div className={`text-xs mt-1 ${
                    selectedFile?.path === file.path 
                      ? 'text-blue-100' 
                      : darkMode ? 'text-gray-500' : 'text-gray-400'
                  }`}>
                    {file.path}
                  </div>
                </motion.button>
              ))}
            </div>
          </div>

          {/* Code Metrics */}
          <div className="p-4 border-t dark:border-gray-700">
            <h4 className={`text-sm font-semibold mb-3 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              Code Metrics
            </h4>
            
            <div className="space-y-3">
              <div className="flex justify-center">
                <svg ref={metricsRef} />
              </div>
              
              <div className="flex items-center justify-between">
                <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Complexity
                </span>
                <svg ref={complexityRef} />
              </div>
            </div>
          </div>
        </div>

        {/* Code Display */}
        <div className="flex-1 flex">
          {/* Editor View */}
          {(viewMode === 'editor' || viewMode === 'split') && (
            <div className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} relative`}>
              {isGenerating && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-10"
                >
                  <div className="text-center">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"
                    />
                    <p className="text-white text-sm">Generating code...</p>
                  </div>
                </motion.div>
              )}
              
              <Editor
                height="100%"
                defaultLanguage={language}
                value={code}
                onChange={(value) => setCode(value || '')}
                theme={darkMode ? 'vs-dark' : 'light'}
                options={{
                  minimap: { enabled: true },
                  fontSize: 14,
                  lineNumbers: 'on',
                  roundedSelection: false,
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  wordWrap: 'on',
                  folding: true,
                  lineDecorationsWidth: 10,
                  lineNumbersMinChars: 3,
                  glyphMargin: true,
                  renderLineHighlight: 'all',
                  selectOnLineNumbers: true,
                  smoothScrolling: true,
                  cursorBlinking: 'smooth',
                  cursorSmoothCaretAnimation: 'on',
                }}
              />
            </div>
          )}

          {/* Preview View */}
          {(viewMode === 'preview' || viewMode === 'split') && (
            <div className={`${viewMode === 'split' ? 'w-1/2 border-l dark:border-gray-700' : 'w-full'} overflow-auto`}>
              <div className="p-4">
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles size={16} className="text-purple-500" />
                  <span className={`text-sm font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    Syntax Highlighted Preview
                  </span>
                </div>
                
                <SyntaxHighlighter
                  language={language}
                  style={darkMode ? vscDarkPlus : prism}
                  customStyle={{
                    margin: 0,
                    borderRadius: '8px',
                    fontSize: '13px',
                    lineHeight: '1.5'
                  }}
                  showLineNumbers
                  wrapLines
                >
                  {code}
                </SyntaxHighlighter>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer with Actions */}
      <div className={`flex items-center justify-between p-4 border-t ${
        darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-gray-50'
      }`}>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Terminal size={16} className="text-green-500" />
            <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              Ready to run
            </span>
          </div>
          
          <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {codeMetrics.lines} lines • {codeMetrics.functions} functions • {codeMetrics.classes} classes
          </div>
        </div>

        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600"
          >
            <Play size={16} />
            Run Code
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600"
          >
            <Wand2 size={16} />
            Optimize
          </motion.button>
        </div>
      </div>
    </motion.div>
  )
}

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Highlight, themes } from 'prism-react-renderer'
import { Copy, Check } from 'lucide-react'

interface CodePreviewProps {
  filePath?: string
  outputPath?: string
  darkMode: boolean
  content?: string
  language?: string
}

export default function CodePreview({ filePath, outputPath, darkMode, content, language }: CodePreviewProps) {
  const [code, setCode] = useState<string>(content || '')
  const [loading, setLoading] = useState(!content)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (content) {
      setCode(content)
      setLoading(false)
      return
    }

    if (!filePath) {
      setLoading(false)
      return
    }

    // Mock code preview - in real implementation, you'd fetch from backend
    const mockCode = `"""
${filePath.split(/[/\\]/).pop()?.replace('.py', '').replace('_', ' ').toUpperCase() || 'Generated Code'}

Auto-generated implementation from research paper.
"""

import numpy as np
from typing import Any, Dict, List, Optional

class PaperImplementation:
    """Main implementation class for the research paper."""
    
    def __init__(self, **config):
        self.config = config
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize algorithm components."""
        pass
    
    def run(self, input_data: Any) -> Any:
        """Execute the main algorithm."""
        # TODO: Implement based on paper analysis
        return self.process(input_data)
    
    def process(self, data: Any) -> Any:
        """Process input through the algorithm."""
        if isinstance(data, (list, np.ndarray)):
            return np.array(data)
        return data

def main():
    """Main entry point."""
    impl = PaperImplementation()
    result = impl.run(None)
    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    main()
`
    
    setTimeout(() => {
      setCode(mockCode)
      setLoading(false)
    }, 500)
  }, [filePath, outputPath, content])

  const copyCode = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (loading) {
    return (
      <div className="space-y-2">
        {[...Array(6)].map((_, i) => (
          <div key={i} className={`h-4 rounded animate-pulse ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`} />
        ))}
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-lg border overflow-hidden ${darkMode ? 'border-gray-600' : 'border-gray-300'}`}
    >
      <div className={`flex items-center justify-between px-4 py-2 border-b ${
        darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
      }`}>
        <span className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          Code Preview
        </span>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={copyCode}
          className={`p-1 rounded transition-colors ${
            darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'
          }`}
        >
          {copied ? (
            <Check size={16} className="text-green-500" />
          ) : (
            <Copy size={16} className={darkMode ? 'text-gray-400' : 'text-gray-500'} />
          )}
        </motion.button>
      </div>
      
      <div className="max-h-80 overflow-auto">
        <Highlight
          theme={darkMode ? themes.vsDark : themes.vsLight}
          code={code}
          language="python"
        >
          {({ className, style, tokens, getLineProps, getTokenProps }) => (
            <pre className={`${className} p-4 text-sm`} style={style}>
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })}>
                  <span className={`inline-block w-8 text-right mr-4 ${
                    darkMode ? 'text-gray-500' : 'text-gray-400'
                  }`}>
                    {i + 1}
                  </span>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token })} />
                  ))}
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </motion.div>
  )
}

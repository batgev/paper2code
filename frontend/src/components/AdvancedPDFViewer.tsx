import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Document, Page, pdfjs } from 'react-pdf'
import { 
  ZoomIn, ZoomOut, RotateCw, Download, Search, 
  ChevronLeft, ChevronRight, Maximize, Eye,
  BookOpen, FileText, Layers, Target
} from 'lucide-react'
import * as d3 from 'd3'

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`

interface AdvancedPDFViewerProps {
  file: File | null
  darkMode: boolean
  onTextExtracted?: (text: string) => void
  highlightTerms?: string[]
  extractionProgress?: { phase: string, progress: number }
}

export default function AdvancedPDFViewer({ 
  file, 
  darkMode, 
  onTextExtracted, 
  highlightTerms = [],
  extractionProgress 
}: AdvancedPDFViewerProps) {
  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(1)
  const [scale, setScale] = useState<number>(1.2)
  const [rotation, setRotation] = useState<number>(0)
  const [loading, setLoading] = useState<boolean>(false)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [fullscreen, setFullscreen] = useState<boolean>(false)
  const [extractedText, setExtractedText] = useState<string>('')
  const [viewMode, setViewMode] = useState<'single' | 'double' | 'scroll'>('single')
  
  const containerRef = useRef<HTMLDivElement>(null)
  const highlightRef = useRef<SVGSVGElement>(null)

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages)
    setLoading(false)
    
    // Real text extraction would happen here via backend API
    if (file && onTextExtracted) {
      // In a real implementation, this would call the backend
      // For now, we'll let the backend handle text extraction
      onTextExtracted(`PDF loaded: ${file.name} with ${numPages} pages`)
    }
  }

  // D3.js visualization for extraction progress
  useEffect(() => {
    if (!extractionProgress || !highlightRef.current) return

    const svg = d3.select(highlightRef.current)
    svg.selectAll("*").remove()

    // Create animated progress visualization
    const width = 300
    const height = 100
    
    svg.attr("width", width).attr("height", height)

    // Background
    svg.append("rect")
       .attr("width", width)
       .attr("height", height)
       .attr("fill", darkMode ? "#1f2937" : "#f9fafb")
       .attr("rx", 8)

    // Progress bar
    const progressWidth = (extractionProgress.progress / 100) * (width - 20)
    
    svg.append("rect")
       .attr("x", 10)
       .attr("y", height / 2 - 10)
       .attr("width", 0)
       .attr("height", 20)
       .attr("fill", "#3b82f6")
       .attr("rx", 10)
       .transition()
       .duration(500)
       .attr("width", progressWidth)

    // Phase text
    svg.append("text")
       .attr("x", width / 2)
       .attr("y", 30)
       .attr("text-anchor", "middle")
       .attr("fill", darkMode ? "#ffffff" : "#374151")
       .attr("font-size", "14px")
       .attr("font-weight", "bold")
       .text(extractionProgress.phase)

    // Percentage
    svg.append("text")
       .attr("x", width / 2)
       .attr("y", height / 2 + 5)
       .attr("text-anchor", "middle")
       .attr("fill", darkMode ? "#d1d5db" : "#6b7280")
       .attr("font-size", "12px")
       .text(`${extractionProgress.progress}%`)

  }, [extractionProgress, darkMode])

  // Highlight search terms on PDF
  useEffect(() => {
    if (!highlightTerms.length || !containerRef.current) return

    // Create highlight overlays using D3
    const container = d3.select(containerRef.current)
    
    // Remove existing highlights
    container.selectAll(".highlight-overlay").remove()

    // Add new highlights with animation
    highlightTerms.forEach((term, index) => {
      // Simulate finding term positions (in real app, would use PDF.js text layer)
      const highlights = [
        { x: 100 + index * 50, y: 200 + index * 30, width: 80, height: 20 },
        { x: 150 + index * 40, y: 300 + index * 25, width: 60, height: 18 },
      ]

      highlights.forEach((highlight, hIndex) => {
        container.append("div")
                 .attr("class", "highlight-overlay")
                 .style("position", "absolute")
                 .style("left", `${highlight.x}px`)
                 .style("top", `${highlight.y}px`)
                 .style("width", `${highlight.width}px`)
                 .style("height", `${highlight.height}px`)
                 .style("background", "#fbbf24")
                 .style("opacity", 0)
                 .style("border-radius", "4px")
                 .style("pointer-events", "none")
                 .style("z-index", 10)
                 .transition()
                 .delay(index * 200 + hIndex * 100)
                 .duration(300)
                 .style("opacity", 0.3)
      })
    })
  }, [highlightTerms])

  const goToPrevPage = () => setPageNumber(prev => Math.max(prev - 1, 1))
  const goToNextPage = () => setPageNumber(prev => Math.min(prev + 1, numPages))
  const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0))
  const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5))
  const rotate = () => setRotation(prev => (prev + 90) % 360)

  if (!file) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`h-96 rounded-xl border-2 border-dashed ${
          darkMode ? 'border-gray-600 bg-gray-800' : 'border-gray-300 bg-gray-50'
        } flex flex-col items-center justify-center text-center p-8`}
      >
        <motion.div
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <FileText size={48} className={darkMode ? 'text-gray-400' : 'text-gray-400'} />
        </motion.div>
        <h3 className={`text-lg font-semibold mt-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
          Upload a PDF to begin
        </h3>
        <p className={`text-sm mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          Interactive PDF viewer with live extraction visualization
        </p>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-2xl overflow-hidden ${
        fullscreen ? 'fixed inset-4 z-50' : ''
      }`}
    >
      {/* Header Controls */}
      <div className={`flex items-center justify-between p-4 border-b ${
        darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-gray-50'
      }`}>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <BookOpen size={20} className="text-blue-500" />
            <span className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {file.name}
            </span>
          </div>
          
          {/* View Mode Selector */}
          <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            {(['single', 'double', 'scroll'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1 rounded text-sm transition-all ${
                  viewMode === mode
                    ? 'bg-blue-500 text-white shadow-md'
                    : darkMode ? 'text-gray-300 hover:text-white' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Search */}
          <div className="relative">
            <Search size={16} className={`absolute left-3 top-2.5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
            <input
              type="text"
              placeholder="Search in PDF..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={`pl-10 pr-4 py-2 w-48 rounded-lg text-sm ${
                darkMode 
                  ? 'bg-gray-700 border-gray-600 text-white' 
                  : 'bg-white border-gray-300 text-gray-900'
              } border focus:ring-2 focus:ring-blue-500`}
            />
          </div>

          {/* Controls */}
          <div className="flex items-center gap-1 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={zoomOut}
              className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
            >
              <ZoomOut size={16} />
            </motion.button>
            
            <span className={`px-2 text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {Math.round(scale * 100)}%
            </span>
            
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={zoomIn}
              className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
            >
              <ZoomIn size={16} />
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={rotate}
              className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
            >
              <RotateCw size={16} />
            </motion.button>
            
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setFullscreen(!fullscreen)}
              className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
            >
              <Maximize size={16} />
            </motion.button>
          </div>
        </div>
      </div>

      {/* Progress Visualization */}
      {extractionProgress && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className={`p-4 border-b ${darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-blue-50'}`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Target size={20} className="text-blue-500" />
              <span className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                Extracting: {extractionProgress.phase}
              </span>
            </div>
            <svg ref={highlightRef} />
          </div>
        </motion.div>
      )}

      {/* PDF Content */}
      <div className="relative" ref={containerRef}>
        {/* Navigation */}
        <div className={`flex items-center justify-between p-4 border-b ${
          darkMode ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <div className="flex items-center gap-2">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={goToPrevPage}
              disabled={pageNumber <= 1}
              className={`p-2 rounded ${
                pageNumber <= 1 
                  ? 'opacity-50 cursor-not-allowed' 
                  : 'hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <ChevronLeft size={20} />
            </motion.button>
            
            <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              Page {pageNumber} of {numPages}
            </span>
            
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={goToNextPage}
              disabled={pageNumber >= numPages}
              className={`p-2 rounded ${
                pageNumber >= numPages 
                  ? 'opacity-50 cursor-not-allowed' 
                  : 'hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <ChevronRight size={20} />
            </motion.button>
          </div>

          <div className="flex items-center gap-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-3 py-1 bg-blue-500 text-white rounded-lg text-sm hover:bg-blue-600"
            >
              <Download size={14} className="mr-1" />
              Download
            </motion.button>
          </div>
        </div>

        {/* PDF Display */}
        <div className="flex justify-center p-6 min-h-96 bg-gray-100 dark:bg-gray-900">
          <AnimatePresence mode="wait">
            <motion.div
              key={pageNumber}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
              className="relative"
            >
              <Document
                file={file}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={(error) => {
                  console.error('PDF load error:', error)
                  setLoading(false)
                }}
                loading={
                  <div className="flex items-center justify-center h-96">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full"
                    />
                  </div>
                }
              >
                {viewMode === 'single' && (
                  <Page
                    pageNumber={pageNumber}
                    scale={scale}
                    rotate={rotation}
                    className="shadow-2xl rounded-lg overflow-hidden"
                  />
                )}
                
                {viewMode === 'double' && (
                  <div className="flex gap-4">
                    <Page
                      pageNumber={pageNumber}
                      scale={scale * 0.8}
                      rotate={rotation}
                      className="shadow-xl rounded-lg overflow-hidden"
                    />
                    {pageNumber < numPages && (
                      <Page
                        pageNumber={pageNumber + 1}
                        scale={scale * 0.8}
                        rotate={rotation}
                        className="shadow-xl rounded-lg overflow-hidden"
                      />
                    )}
                  </div>
                )}
                
                {viewMode === 'scroll' && (
                  <div className="space-y-4">
                    {Array.from({ length: Math.min(3, numPages - pageNumber + 1) }, (_, i) => (
                      <Page
                        key={pageNumber + i}
                        pageNumber={pageNumber + i}
                        scale={scale * 0.9}
                        rotate={rotation}
                        className="shadow-xl rounded-lg overflow-hidden"
                      />
                    ))}
                  </div>
                )}
              </Document>

              {/* Highlight overlays will be added here by D3 */}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Extraction Visualization */}
        {extractionProgress && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`absolute bottom-4 right-4 p-4 rounded-xl shadow-lg ${
              darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
            }`}
          >
            <div className="flex items-center gap-3">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="w-6 h-6"
              >
                <Layers size={24} className="text-blue-500" />
              </motion.div>
              <div>
                <div className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {extractionProgress.phase}
                </div>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {extractionProgress.progress}% complete
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Fullscreen overlay */}
      {fullscreen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={() => setFullscreen(false)}
        />
      )}
    </motion.div>
  )
}

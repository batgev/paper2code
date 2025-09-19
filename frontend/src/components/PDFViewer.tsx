import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Document, Page, pdfjs } from 'react-pdf'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, FileText, Eye } from 'lucide-react'

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`

interface PDFViewerProps {
  file: File | null
  darkMode: boolean
  onTextExtracted?: (text: string) => void
}

export default function PDFViewer({ file, darkMode, onTextExtracted }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number>(0)
  const [pageNumber, setPageNumber] = useState<number>(1)
  const [scale, setScale] = useState<number>(1.0)
  const [loading, setLoading] = useState<boolean>(false)
  const [extractedText, setExtractedText] = useState<string>('')
  const [showText, setShowText] = useState<boolean>(false)

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages)
    setLoading(false)
  }

  function onDocumentLoadError(error: Error) {
    console.error('PDF load error:', error)
    setLoading(false)
  }

  const goToPrevPage = () => {
    setPageNumber(prev => Math.max(prev - 1, 1))
  }

  const goToNextPage = () => {
    setPageNumber(prev => Math.min(prev + 1, numPages))
  }

  const zoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 3.0))
  }

  const zoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.5))
  }

  const extractText = async () => {
    if (!file) return
    
    setLoading(true)
    try {
      // Simulate text extraction (in real app, this would call the backend)
      const mockText = `
# ${file.name.replace('.pdf', '')}

## Abstract
This paper presents a novel approach to neural network architectures...

## Introduction
Recent advances in deep learning have shown...

## Method
We propose a new attention mechanism that...

## Experiments
Our experiments demonstrate...

## Results
The proposed method achieves state-of-the-art performance...

## Conclusion
In this work, we have presented...
      `.trim()
      
      setExtractedText(mockText)
      onTextExtracted?.(mockText)
      setShowText(true)
    } catch (error) {
      console.error('Text extraction error:', error)
    } finally {
      setLoading(false)
    }
  }

  if (!file) {
    return (
      <div className={`flex items-center justify-center h-96 border-2 border-dashed rounded-xl ${
        darkMode ? 'border-gray-600 bg-gray-800/50' : 'border-gray-300 bg-gray-50'
      }`}>
        <div className="text-center">
          <FileText size={48} className={darkMode ? 'text-gray-500' : 'text-gray-400'} />
          <p className={`mt-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Upload a PDF to preview
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} rounded-xl border overflow-hidden`}>
      {/* PDF Viewer Header */}
      <div className={`flex items-center justify-between p-4 border-b ${darkMode ? 'border-gray-700 bg-gray-700' : 'border-gray-200 bg-gray-50'}`}>
        <div className="flex items-center gap-3">
          <FileText size={20} className={darkMode ? 'text-gray-400' : 'text-gray-600'} />
          <div>
            <h3 className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              {file.name}
            </h3>
            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={extractText}
            disabled={loading}
            className={`px-3 py-2 rounded-lg text-sm transition-colors ${
              darkMode 
                ? 'bg-blue-600 hover:bg-blue-500 text-white' 
                : 'bg-blue-600 hover:bg-blue-500 text-white'
            } disabled:opacity-50`}
          >
            <Eye size={16} className="mr-2" />
            {loading ? 'Extracting...' : 'Extract Text'}
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowText(!showText)}
            className={`px-3 py-2 rounded-lg text-sm transition-colors ${
              showText 
                ? darkMode ? 'bg-green-600 text-white' : 'bg-green-600 text-white'
                : darkMode ? 'bg-gray-600 hover:bg-gray-500 text-gray-300' : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
            }`}
          >
            {showText ? 'Hide Text' : 'Show Text'}
          </motion.button>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-0">
        {/* PDF Preview */}
        <div className={`p-4 ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <div className="flex items-center justify-between mb-4">
            <h4 className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              PDF Preview
            </h4>
            
            <div className="flex items-center gap-2">
              {/* Zoom Controls */}
              <div className="flex items-center gap-1">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={zoomOut}
                  className={`p-1 rounded ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
                >
                  <ZoomOut size={16} />
                </motion.button>
                <span className={`text-sm px-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  {Math.round(scale * 100)}%
                </span>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={zoomIn}
                  className={`p-1 rounded ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
                >
                  <ZoomIn size={16} />
                </motion.button>
              </div>
              
              {/* Page Navigation */}
              <div className="flex items-center gap-1">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={goToPrevPage}
                  disabled={pageNumber <= 1}
                  className={`p-1 rounded disabled:opacity-50 ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
                >
                  <ChevronLeft size={16} />
                </motion.button>
                <span className={`text-sm px-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  {pageNumber} / {numPages}
                </span>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={goToNextPage}
                  disabled={pageNumber >= numPages}
                  className={`p-1 rounded disabled:opacity-50 ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
                >
                  <ChevronRight size={16} />
                </motion.button>
              </div>
            </div>
          </div>
          
          <div className="border rounded-lg overflow-hidden">
            <Document
              file={file}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={
                <div className={`flex items-center justify-center h-96 ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                    <p className={darkMode ? 'text-gray-300' : 'text-gray-600'}>Loading PDF...</p>
                  </div>
                </div>
              }
            >
              <Page
                pageNumber={pageNumber}
                scale={scale}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
            </Document>
          </div>
        </div>

        {/* Extracted Text */}
        {showText && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`p-4 border-l ${darkMode ? 'bg-gray-900 border-gray-700' : 'bg-gray-50 border-gray-200'}`}
          >
            <h4 className={`font-medium mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Extracted Text
            </h4>
            
            <div className={`h-96 overflow-y-auto p-4 rounded-lg border text-sm ${
              darkMode ? 'bg-gray-800 border-gray-600 text-gray-300' : 'bg-white border-gray-300 text-gray-700'
            }`}>
              {extractedText ? (
                <pre className="whitespace-pre-wrap font-mono">{extractedText}</pre>
              ) : (
                <div className={`text-center py-8 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  Click "Extract Text" to see PDF content
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

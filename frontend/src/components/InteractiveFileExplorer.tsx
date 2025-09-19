import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Folder, 
  File, 
  Code, 
  Download, 
  Eye, 
  Search, 
  ChevronRight, 
  ChevronDown,
  FileText,
  Settings,
  Package,
  Image,
  Archive
} from 'lucide-react'
import axios from 'axios'
import CodePreview from './CodePreview'

interface FileItem {
  path: string
  name: string
  size: number
  modified: number
  extension: string
  type: 'code' | 'other'
}

interface FileExplorerProps {
  taskId: string
  darkMode: boolean
}

interface FileNode {
  name: string
  path: string
  type: 'file' | 'folder'
  children?: FileNode[]
  file?: FileItem
}

export default function InteractiveFileExplorer({ taskId, darkMode }: FileExplorerProps) {
  const [files, setFiles] = useState<FileItem[]>([])
  const [fileTree, setFileTree] = useState<FileNode[]>([])
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
  const [selectedFile, setSelectedFile] = useState<FileItem | null>(null)
  const [fileContent, setFileContent] = useState<string>('')
  const [searchTerm, setSearchTerm] = useState('')
  const [loading, setLoading] = useState(true)
  const [viewMode, setViewMode] = useState<'tree' | 'list'>('tree')

  useEffect(() => {
    if (taskId) {
      loadFiles()
    }
  }, [taskId])

  useEffect(() => {
    if (files.length > 0) {
      buildFileTree()
    }
  }, [files])

  const loadFiles = async () => {
    try {
      setLoading(true)
      const response = await axios.get(`/api/files/${taskId}`)
      setFiles(response.data.files || [])
    } catch (error) {
      console.error('Failed to load files:', error)
    } finally {
      setLoading(false)
    }
  }

  const buildFileTree = () => {
    const tree: FileNode[] = []
    const folderMap = new Map<string, FileNode>()

    files.forEach(file => {
      const pathParts = file.path.split('/')
      let currentPath = ''
      
      pathParts.forEach((part, index) => {
        const parentPath = currentPath
        currentPath = currentPath ? `${currentPath}/${part}` : part
        
        if (index === pathParts.length - 1) {
          // This is a file
          const fileNode: FileNode = {
            name: part,
            path: currentPath,
            type: 'file',
            file
          }
          
          if (parentPath) {
            const parent = folderMap.get(parentPath)
            if (parent && parent.children) {
              parent.children.push(fileNode)
            }
          } else {
            tree.push(fileNode)
          }
        } else {
          // This is a folder
          if (!folderMap.has(currentPath)) {
            const folderNode: FileNode = {
              name: part,
              path: currentPath,
              type: 'folder',
              children: []
            }
            
            folderMap.set(currentPath, folderNode)
            
            if (parentPath) {
              const parent = folderMap.get(parentPath)
              if (parent && parent.children) {
                parent.children.push(folderNode)
              }
            } else {
              tree.push(folderNode)
            }
          }
        }
      })
    })

    // Sort tree (folders first, then files)
    const sortNodes = (nodes: FileNode[]) => {
      return nodes.sort((a, b) => {
        if (a.type !== b.type) {
          return a.type === 'folder' ? -1 : 1
        }
        return a.name.localeCompare(b.name)
      }).map(node => ({
        ...node,
        children: node.children ? sortNodes(node.children) : undefined
      }))
    }

    setFileTree(sortNodes(tree))
  }

  const toggleFolder = (path: string) => {
    setExpandedFolders(prev => {
      const newSet = new Set(prev)
      if (newSet.has(path)) {
        newSet.delete(path)
      } else {
        newSet.add(path)
      }
      return newSet
    })
  }

  const viewFile = async (file: FileItem) => {
    try {
      setSelectedFile(file)
      const response = await axios.get(`/api/view/${taskId}?file_path=${encodeURIComponent(file.path)}`)
      setFileContent(response.data.content)
    } catch (error) {
      console.error('Failed to load file content:', error)
      setFileContent('Error loading file content')
    }
  }

  const downloadProject = async () => {
    try {
      const response = await axios.get(`/api/download/${taskId}`, {
        responseType: 'blob'
      })
      
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `project_${taskId}.zip`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download project:', error)
    }
  }

  const getFileIcon = (file: FileItem) => {
    const ext = file.extension.toLowerCase()
    if (['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h'].includes(ext)) {
      return <Code size={16} className="text-blue-500" />
    }
    if (['.json', '.yaml', '.yml', '.xml', '.toml'].includes(ext)) {
      return <Settings size={16} className="text-yellow-500" />
    }
    if (['.md', '.txt', '.rst'].includes(ext)) {
      return <FileText size={16} className="text-green-500" />
    }
    if (['.png', '.jpg', '.jpeg', '.gif', '.svg'].includes(ext)) {
      return <Image size={16} className="text-purple-500" />
    }
    if (['.zip', '.tar', '.gz'].includes(ext)) {
      return <Archive size={16} className="text-orange-500" />
    }
    if (ext === '.pdf') {
      return <FileText size={16} className="text-red-500" />
    }
    return <File size={16} className={darkMode ? 'text-gray-400' : 'text-gray-500'} />
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString()
  }

  const filteredFiles = files.filter(file =>
    file.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    file.path.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const renderTreeNode = (node: FileNode, level = 0) => {
    const isExpanded = expandedFolders.has(node.path)
    
    return (
      <div key={node.path} className="select-none">
        <motion.div
          className={`flex items-center gap-2 px-2 py-1 rounded cursor-pointer hover:bg-opacity-50 ${
            darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
          } ${selectedFile?.path === node.path ? (darkMode ? 'bg-gray-700' : 'bg-gray-100') : ''}`}
          style={{ marginLeft: level * 20 }}
          onClick={() => {
            if (node.type === 'folder') {
              toggleFolder(node.path)
            } else if (node.file) {
              viewFile(node.file)
            }
          }}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
        >
          {node.type === 'folder' ? (
            <>
              {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              <Folder size={16} className="text-blue-500" />
              <span className={darkMode ? 'text-white' : 'text-gray-900'}>{node.name}</span>
              <span className={`text-xs ml-auto ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {node.children?.length || 0} items
              </span>
            </>
          ) : (
            <>
              <div style={{ width: 16 }} /> {/* Spacer for alignment */}
              {getFileIcon(node.file!)}
              <span className={darkMode ? 'text-white' : 'text-gray-900'}>{node.name}</span>
              <span className={`text-xs ml-auto ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {formatFileSize(node.file!.size)}
              </span>
            </>
          )}
        </motion.div>
        
        <AnimatePresence>
          {node.type === 'folder' && isExpanded && node.children && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
            >
              {node.children.map(child => renderTreeNode(child, level + 1))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    )
  }

  const renderListItem = (file: FileItem) => (
    <motion.div
      key={file.path}
      className={`flex items-center justify-between p-3 rounded-lg cursor-pointer group ${
        darkMode ? 'hover:bg-gray-700 bg-gray-800' : 'hover:bg-gray-50 bg-white'
      } ${selectedFile?.path === file.path ? (darkMode ? 'bg-gray-700' : 'bg-gray-100') : ''}`}
      onClick={() => viewFile(file)}
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
    >
      <div className="flex items-center gap-3">
        {getFileIcon(file)}
        <div>
          <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            {file.name}
          </div>
          <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            {file.path}
          </div>
        </div>
      </div>
      <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        <div>{formatFileSize(file.size)}</div>
        <div>{formatDate(file.modified)}</div>
      </div>
    </motion.div>
  )

  if (loading) {
    return (
      <div className={`p-6 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="animate-pulse space-y-4">
          <div className={`h-4 rounded ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}></div>
          <div className={`h-4 rounded w-3/4 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}></div>
          <div className={`h-4 rounded w-1/2 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}></div>
        </div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* File Explorer */}
      <div className={`rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} p-6`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Project Files ({files.length})
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode(viewMode === 'tree' ? 'list' : 'tree')}
              className={`px-3 py-1 rounded text-sm ${
                darkMode ? 'bg-gray-700 text-white' : 'bg-gray-100 text-gray-700'
              }`}
            >
              {viewMode === 'tree' ? 'List' : 'Tree'}
            </button>
            <motion.button
              onClick={downloadProject}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Download size={16} />
              Download ZIP
            </motion.button>
          </div>
        </div>

        {/* Search */}
        <div className="relative mb-4">
          <Search size={16} className={`absolute left-3 top-3 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
          <input
            type="text"
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={`w-full pl-10 pr-4 py-2 rounded-lg border ${
              darkMode 
                ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
            } focus:ring-2 focus:ring-blue-500 focus:border-transparent`}
          />
        </div>

        {/* File List */}
        <div className="max-h-96 overflow-y-auto space-y-1">
          {searchTerm ? (
            // Filtered list view when searching
            filteredFiles.map(file => renderListItem(file))
          ) : viewMode === 'tree' ? (
            // Tree view
            fileTree.map(node => renderTreeNode(node))
          ) : (
            // List view
            files.map(file => renderListItem(file))
          )}
        </div>
      </div>

      {/* File Viewer */}
      <div className={`rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} p-6`}>
        {selectedFile ? (
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                {getFileIcon(selectedFile)}
                <h3 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {selectedFile.name}
                </h3>
              </div>
              <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                {formatFileSize(selectedFile.size)}
              </div>
            </div>
            
            <div className={`text-sm mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {selectedFile.path}
            </div>

            <div className="max-h-96 overflow-y-auto">
              <CodePreview
                content={fileContent}
                language={selectedFile.extension.slice(1)}
                darkMode={darkMode}
              />
            </div>
          </div>
        ) : (
          <div className={`text-center py-12 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            <Eye size={48} className="mx-auto mb-4 opacity-50" />
            <p>Select a file to view its content</p>
          </div>
        )}
      </div>
    </div>
  )
}

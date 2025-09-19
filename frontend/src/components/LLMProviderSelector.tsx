import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Key, Server, AlertCircle, CheckCircle, RefreshCw } from 'lucide-react'
import axios from 'axios'

interface Provider {
  name: string
  display_name: string
  available: boolean
  requires_api_key: boolean
  error?: string
}

interface Model {
  name: string
  description?: string
  size?: string
}

interface LLMProviderSelectorProps {
  selectedProvider: string
  selectedModel: string
  onProviderChange: (provider: string) => void
  onModelChange: (model: string) => void
  onApiKeyChange: (provider: string, key: string) => void
  darkMode: boolean
  disabled?: boolean
}

export default function LLMProviderSelector({
  selectedProvider,
  selectedModel,
  onProviderChange,
  onModelChange,
  onApiKeyChange,
  darkMode,
  disabled = false
}: LLMProviderSelectorProps) {
  const [providers, setProviders] = useState<Provider[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [apiKeys, setApiKeys] = useState<{[key: string]: string}>({})
  const [loading, setLoading] = useState(false)
  const [showApiKeyInput, setShowApiKeyInput] = useState(false)

  useEffect(() => {
    loadProviders()
  }, [])

  useEffect(() => {
    if (selectedProvider) {
      loadModels(selectedProvider)
    }
  }, [selectedProvider])

  const loadProviders = async () => {
    try {
      const { data } = await axios.get('/api/llm/providers')
      setProviders(data.providers || [])
    } catch (e) {
      console.error('Failed to load providers:', e)
    }
  }

  const loadModels = async (provider: string) => {
    setLoading(true)
    try {
      const { data } = await axios.get(`/api/llm/models/${provider}`)
      setModels(data.models || [])
      
      // Auto-select first model if current selection not available
      if (data.models?.length && !data.models.find((m: Model) => m.name === selectedModel)) {
        onModelChange(data.models[0].name)
      }
    } catch (e) {
      console.error('Failed to load models:', e)
      setModels([])
    } finally {
      setLoading(false)
    }
  }

  const handleProviderSelect = (provider: string) => {
    onProviderChange(provider)
    
    const providerInfo = providers.find(p => p.name === provider)
    if (providerInfo?.requires_api_key) {
      setShowApiKeyInput(true)
    } else {
      setShowApiKeyInput(false)
    }
  }

  const handleApiKeySubmit = (provider: string) => {
    const key = apiKeys[provider]
    if (key) {
      onApiKeyChange(provider, key)
      setShowApiKeyInput(false)
    }
  }

  const selectedProviderInfo = providers.find(p => p.name === selectedProvider)

  return (
    <div className="space-y-4">
      {/* Provider Selection */}
      <div>
        <label className={`block text-sm font-medium mb-3 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          LLM Provider
        </label>
        <div className="grid gap-3">
          {providers.map((provider) => (
            <motion.button
              key={provider.name}
              whileHover={{ scale: disabled ? 1 : 1.02 }}
              whileTap={{ scale: disabled ? 1 : 0.98 }}
              onClick={() => !disabled && handleProviderSelect(provider.name)}
              disabled={disabled || !provider.available}
              className={`p-4 rounded-xl border-2 text-left transition-all ${
                selectedProvider === provider.name
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : provider.available
                    ? `${darkMode ? 'border-gray-600 hover:border-gray-500' : 'border-gray-300 hover:border-gray-400'}`
                    : `${darkMode ? 'border-gray-700 bg-gray-800/50' : 'border-gray-200 bg-gray-100'} opacity-50`
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${
                    selectedProvider === provider.name 
                      ? 'bg-blue-500 text-white' 
                      : darkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-100 text-gray-500'
                  }`}>
                    {provider.name === 'ollama' ? <Server size={16} /> : <Brain size={16} />}
                  </div>
                  <div>
                    <div className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      {provider.display_name}
                    </div>
                    <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      {provider.requires_api_key ? 'Requires API key' : 'No API key needed'}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  {provider.requires_api_key && (
                    <Key size={14} className={darkMode ? 'text-gray-500' : 'text-gray-400'} />
                  )}
                  {provider.available ? (
                    <CheckCircle size={16} className="text-green-500" />
                  ) : (
                    <AlertCircle size={16} className="text-red-500" />
                  )}
                </div>
              </div>
              
              {!provider.available && provider.error && (
                <div className={`mt-2 text-xs ${darkMode ? 'text-red-400' : 'text-red-600'}`}>
                  {provider.error}
                </div>
              )}
            </motion.button>
          ))}
        </div>
      </div>

      {/* API Key Input */}
      <AnimatePresence>
        {showApiKeyInput && selectedProviderInfo?.requires_api_key && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className={`p-4 rounded-xl border ${darkMode ? 'border-yellow-600 bg-yellow-900/20' : 'border-yellow-300 bg-yellow-50'}`}
          >
            <div className="flex items-center gap-2 mb-3">
              <Key size={16} className={darkMode ? 'text-yellow-400' : 'text-yellow-600'} />
              <span className={`font-medium ${darkMode ? 'text-yellow-400' : 'text-yellow-700'}`}>
                API Key Required for {selectedProviderInfo.display_name}
              </span>
            </div>
            
            <div className="flex gap-3">
              <input
                type="password"
                placeholder={`Enter ${selectedProvider.toUpperCase()} API key`}
                className={`flex-1 px-3 py-2 rounded-lg border transition-colors ${
                  darkMode 
                    ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400 focus:border-blue-500' 
                    : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-blue-500'
                } focus:outline-none focus:ring-2 focus:ring-blue-500/20`}
                value={apiKeys[selectedProvider] || ''}
                onChange={(e) => setApiKeys(prev => ({...prev, [selectedProvider]: e.target.value}))}
              />
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => handleApiKeySubmit(selectedProvider)}
                disabled={!apiKeys[selectedProvider]}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors disabled:opacity-50"
              >
                Save
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Model Selection */}
      {selectedProvider && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <label className={`text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              Model
            </label>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => loadModels(selectedProvider)}
              disabled={disabled || loading}
              className={`p-1 rounded transition-colors ${
                darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
              }`}
            >
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            </motion.button>
          </div>
          
          <select
            className={`w-full px-4 py-3 rounded-lg border transition-colors ${
              darkMode 
                ? 'bg-gray-700 border-gray-600 text-white focus:border-blue-500' 
                : 'bg-white border-gray-300 text-gray-900 focus:border-blue-500'
            } focus:outline-none focus:ring-2 focus:ring-blue-500/20`}
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={disabled || loading}
          >
            {loading && <option>Loading models...</option>}
            {!loading && models.length === 0 && <option>No models available</option>}
            {models.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name} {model.size && `(${model.size})`} {model.description && `- ${model.description}`}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  )
}

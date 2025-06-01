"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Loader2, Sparkles } from "lucide-react"

interface Entity {
  text: string
  label: string
  start: number
  end: number
}

const entityColors = {
  PER: "bg-blue-100 text-blue-800 border-blue-200",
  LOC: "bg-green-100 text-green-800 border-green-200",
  ORG: "bg-purple-100 text-purple-800 border-purple-200",
  DATE: "bg-amber-100 text-amber-800 border-amber-200",
}

const entityLabels = {
  PER: "PERSON_NAME",
  LOC: "LOCATION",
  ORG: "ORGANIZATION",
  DATE: "DATE",
}

export default function Component() {
  const [inputText, setInputText] = useState("")
  const [entities, setEntities] = useState<Entity[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [hasResult, setHasResult] = useState(false)

  const processText = async () => {
    if (!inputText.trim()) return

    setIsProcessing(true)
    setHasResult(false)
    setEntities([])

    try {
      const response = await fetch("http://localhost:8000/api/ner", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText }),
      })

      if (!response.ok) {
        throw new Error("Failed to process text")
      }

      const data = await response.json()
      setEntities(data.entities)
      setHasResult(true)
    } catch (error) {
      console.error("Error processing text:", error)
      // Xử lý lỗi ở đây nếu cần
    } finally {
      setIsProcessing(false)
    }
  }

  const renderHighlightedText = () => {
    if (!hasResult || entities.length === 0) {
      return <span className="text-muted-foreground">{inputText}</span>
    }

    let lastIndex = 0
    const elements = []

    // Sort entities by start position
    const sortedEntities = [...entities].sort((a, b) => a.start - b.start)

    sortedEntities.forEach((entity, index) => {
      const entityStart = inputText.toLowerCase().indexOf(entity.text.toLowerCase(), lastIndex)

      if (entityStart !== -1) {
        // Add text before entity
        if (entityStart > lastIndex) {
          elements.push(<span key={`text-${index}`}>{inputText.slice(lastIndex, entityStart)}</span>)
        }

        // Add highlighted entity
        elements.push(
          <span
            key={`entity-${index}`}
            className={`px-1 py-0.5 rounded text-sm font-medium ${entityColors[entity.label as keyof typeof entityColors]}`}
          >
            {inputText.slice(entityStart, entityStart + entity.text.length)}
          </span>,
        )

        lastIndex = entityStart + entity.text.length
      }
    })

    // Add remaining text
    if (lastIndex < inputText.length) {
      elements.push(<span key="remaining">{inputText.slice(lastIndex)}</span>)
    }

    return elements
  }

  const handleExample = () => {
    const exampleText =
      "Nguyễn Văn A là sinh viên tại Đại học Bách Khoa Hà Nội, Việt Nam. Vào ngày 15/06/2023, anh ấy đã tốt nghiệp."
    setInputText(exampleText)
    setHasResult(false)
    setEntities([])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 font-['Inter',_'system-ui',_sans-serif]">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2">
            <Sparkles className="w-8 h-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-900 font-['Inter',_'system-ui',_sans-serif] tracking-tight">
              Nhận Diện Thực Thể Tiếng Việt
            </h1>
          </div>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Hệ thống AI nhận diện và phân loại các thực thể trong văn bản tiếng Việt như tên người, địa điểm, tổ chức
          </p>
        </div>

        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">Nhập văn bản cần phân tích</CardTitle>
            <CardDescription>Nhập đoạn văn bản tiếng Việt để nhận diện các thực thể</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Ví dụ: Nguyễn Văn A là sinh viên tại Đại học Bách Khoa Hà Nội, Việt Nam..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="min-h-[120px] text-base font-['Inter',_'system-ui',_sans-serif]"
            />
            <div className="flex gap-2">
              <Button
                onClick={processText}
                disabled={!inputText.trim() || isProcessing}
                className="flex items-center gap-2"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Đang xử lý...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Phân tích
                  </>
                )}
              </Button>
              <Button variant="outline" onClick={handleExample}>
                Dùng ví dụ
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results Section */}
        {(hasResult || isProcessing) && (
          <Card>
            <CardHeader>
              <CardTitle>Kết quả phân tích</CardTitle>
              <CardDescription>Các thực thể được nhận diện và phân loại</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Highlighted Text */}
              <div className="space-y-2">
                <h3 className="font-semibold text-sm text-gray-700">Văn bản với thực thể được đánh dấu:</h3>
                <div className="p-4 bg-gray-50 rounded-lg border text-base leading-relaxed font-['Inter',_'system-ui',_sans-serif]">
                  {isProcessing ? (
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Đang phân tích văn bản...
                    </div>
                  ) : (
                    renderHighlightedText()
                  )}
                </div>
              </div>

              {/* Entity Legend */}
              {hasResult && entities.length > 0 && (
                <div className="space-y-2">
                  <h3 className="font-semibold text-sm text-gray-700">Chú thích:</h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(entityLabels).map(([key, label]) => (
                      <Badge key={key} variant="outline" className={entityColors[key as keyof typeof entityColors]}>
                        {label}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Entity List */}
              {hasResult && entities.length > 0 && (
                <div className="space-y-2">
                  <h3 className="font-semibold text-sm text-gray-700">Danh sách thực thể ({entities.length}):</h3>
                  <div className="grid gap-2">
                    {entities.map((entity, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-white border rounded-lg">
                        <span className="font-medium">{entity.text}</span>
                        <Badge variant="outline" className={entityColors[entity.label as keyof typeof entityColors]}>
                          {entityLabels[entity.label as keyof typeof entityLabels]}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {hasResult && entities.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <Sparkles className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>Không tìm thấy thực thể nào trong văn bản này</p>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

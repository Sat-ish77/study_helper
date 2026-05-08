"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { Upload, X, Loader2 } from "lucide-react";
import { useAppStore } from "@/store/useAppStore";
import { ingestFile, getDocuments, deleteDocument } from "@/lib/api";
import type { Document } from "@/types";

interface UploadingFile {
  file: File;
  status: "uploading" | "processing" | "chunking" | "embedding" | "done" | "error";
  chunks?: number;
}

export default function UploadPage() {
  const { token } = useAppStore();
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDocuments = async () => {
    if (!token) return;
    try {
      const data = await getDocuments(token) as any;
      setDocuments(data.documents || []);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [token]);

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files || !token) return;

    const validFiles = Array.from(files).filter(
      (file) =>
        file.name.endsWith(".pdf") ||
        file.name.endsWith(".docx") ||
        file.name.endsWith(".pptx")
    );

    if (validFiles.length === 0) return;

    const newUploadingFiles = validFiles.map((file) => ({
      file,
      status: "uploading" as const,
    }));
    setUploadingFiles((prev) => [...prev, ...newUploadingFiles]);

    for (const uploadingFile of newUploadingFiles) {
      try {
        const updateStatus = (status: UploadingFile["status"]) => {
          setUploadingFiles((prev) =>
            prev.map((f) =>
              f.file === uploadingFile.file ? { ...f, status } : f
            )
          );
        };

        updateStatus("uploading");
        await new Promise((r) => setTimeout(r, 500));
        updateStatus("processing");
        await new Promise((r) => setTimeout(r, 500));
        updateStatus("chunking");
        await new Promise((r) => setTimeout(r, 500));
        updateStatus("embedding");

        const result = await ingestFile(uploadingFile.file, token);

        updateStatus("done");
        setUploadingFiles((prev) =>
          prev.map((f) =>
            f.file === uploadingFile.file
              ? { ...f, status: "done", chunks: result.chunks_created }
              : f
          )
        );

        fetchDocuments();
      } catch (err) {
        console.error("Failed to upload file:", err);
        setUploadingFiles((prev) =>
          prev.map((f) =>
            f.file === uploadingFile.file ? { ...f, status: "error" } : f
          )
        );
      }
    }

    setTimeout(() => {
      setUploadingFiles((prev) =>
        prev.filter((f) => f.status !== "done" && f.status !== "error")
      );
    }, 5000);
  }, [token]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const onFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    e.target.value = "";
  }, [handleFiles]);

  const handleDelete = async (sourceFile: string) => {
    if (!token) return;
    try {
      await deleteDocument(sourceFile, token);
      fetchDocuments();
    } catch (err) {
      console.error("Failed to delete document:", err);
    }
  };

  const getFileIcon = (filename: string) => {
    if (filename.endsWith(".pdf")) return "🔴";
    if (filename.endsWith(".docx")) return "🔵";
    if (filename.endsWith(".pptx")) return "🟠";
    return "📄";
  };

  const getStatusText = (status: UploadingFile["status"]) => {
    switch (status) {
      case "uploading":
        return "Uploading...";
      case "processing":
        return "Processing...";
      case "chunking":
        return "Chunking...";
      case "embedding":
        return "Embedding...";
      case "done":
        return "Ready";
      case "error":
        return "Error";
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-serif text-text-primary mb-6">Upload Documents</h1>

      {/* Dropzone */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`
          border-2 border-dashed rounded-xl p-12 text-center cursor-pointer
          transition-all duration-200
          ${isDragOver
            ? "border-accent bg-accent/5"
            : "border-border-subtle hover:border-border-default"
          }
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.pptx"
          onChange={onFileInputChange}
          className="hidden"
        />
        <Upload size={48} className="mx-auto mb-4 text-accent/60" />
        <p className="text-lg text-text-primary mb-2">
          {isDragOver ? "Drop files here..." : "Drop PDF, DOCX, or PPTX here"}
        </p>
        <p className="text-sm text-text-secondary">or click to browse</p>
      </div>

      {/* Uploading Files */}
      {uploadingFiles.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 space-y-3"
        >
          <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider">
            Uploading
          </h2>
          {uploadingFiles.map((uploadingFile, index) => (
            <div
              key={index}
              className="flex items-center gap-3 p-3 rounded-xl bg-background-card border border-border-subtle"
            >
              <span className="text-2xl">{getFileIcon(uploadingFile.file.name)}</span>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-text-primary truncate">{uploadingFile.file.name}</p>
                <p className="text-xs text-text-secondary">
                  {getStatusText(uploadingFile.status)}
                  {uploadingFile.chunks && ` • ${uploadingFile.chunks} chunks`}
                </p>
              </div>
              {uploadingFile.status === "done" ? (
                <span className="text-status-success text-sm">✓</span>
              ) : uploadingFile.status === "error" ? (
                <span className="text-status-error text-sm">✗</span>
              ) : (
                <Loader2 size={16} className="animate-spin text-accent" />
              )}
            </div>
          ))}
        </motion.div>
      )}

      {/* Documents List */}
      <div className="mt-8">
        <h2 className="text-sm font-medium text-text-secondary uppercase tracking-wider mb-4">
          Your Documents ({documents.length})
        </h2>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 size={24} className="animate-spin text-accent" />
          </div>
        ) : documents.length === 0 ? (
          <p className="text-center text-text-secondary py-12">
            No documents uploaded yet. Drop files above to get started.
          </p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {documents.map((doc, index) => (
              <motion.div
                key={doc.source_file}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ borderColor: "rgba(232, 164, 74, 0.25)" }}
                className="group relative p-4 rounded-xl bg-background-card border border-border-subtle transition-all"
              >
                <div className="flex items-start gap-3">
                  <span className="text-2xl">{getFileIcon(doc.source_file)}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-text-primary truncate">{doc.source_file}</p>
                    <p className="text-xs text-text-tertiary uppercase">{doc.filetype}</p>
                  </div>
                </div>

                <button
                  onClick={() => handleDelete(doc.source_file)}
                  className="absolute top-2 right-2 p-1.5 rounded-lg opacity-0 group-hover:opacity-100
                           text-text-tertiary hover:text-status-error hover:bg-white/5 transition-all"
                >
                  <X size={14} />
                </button>
              </motion.div>
            ))}
          </div>
        )}
      </div>

    </div>
  );
}

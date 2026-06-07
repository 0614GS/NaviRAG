import { useRef, useState } from 'react';
import { Upload, Loader2 } from 'lucide-react';
import { apiUpload } from '../api/client';

interface Props {
  onUploaded: () => void;
}

export default function DocumentUpload({ onUploaded }: Props) {
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      await apiUpload('/documents/upload', file);
      onUploaded();
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
      if (inputRef.current) {
        inputRef.current.value = '';
      }
    }
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".md,.markdown,.txt"
        onChange={handleUpload}
        className="hidden"
      />
      <button
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm
                   hover:bg-indigo-700 disabled:opacity-50 transition-colors"
      >
        {uploading ? (
          <Loader2 size={16} className="animate-spin" />
        ) : (
          <Upload size={16} />
        )}
        {uploading ? '上传中…' : '上传文档'}
      </button>
    </>
  );
}

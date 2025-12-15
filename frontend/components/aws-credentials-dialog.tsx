"use client"

import { useState, useEffect } from "react"
import { Settings } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { CheckCircle2 } from "lucide-react"

interface AWSCredentials {
  AWS_ACCESS_KEY_ID: string
  AWS_SECRET_ACCESS_KEY: string
  AWS_SESSION_TOKEN: string
  AWS_REGION: string
}

export function AWSCredentialsDialog() {
  const [open, setOpen] = useState(false)
  const [credentials, setCredentials] = useState<AWSCredentials>({
    AWS_ACCESS_KEY_ID: "",
    AWS_SECRET_ACCESS_KEY: "",
    AWS_SESSION_TOKEN: "",
    AWS_REGION: "",
  })

  // Load credentials from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem("aws_credentials")
    if (stored) {
      try {
        setCredentials(JSON.parse(stored))
      } catch (e) {
        console.error("Failed to parse stored credentials", e)
      }
    }
  }, [])

  // Check if all fields are filled
  const allFieldsFilled = 
    credentials.AWS_ACCESS_KEY_ID.trim() !== "" &&
    credentials.AWS_SECRET_ACCESS_KEY.trim() !== "" &&
    credentials.AWS_SESSION_TOKEN.trim() !== "" &&
    credentials.AWS_REGION.trim() !== ""

  const handleInputChange = (field: keyof AWSCredentials, value: string) => {
    setCredentials((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const handleSave = () => {
    // Save to localStorage
    localStorage.setItem("aws_credentials", JSON.stringify(credentials))
    setOpen(false)
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative h-9 w-9"
          aria-label="AWS Settings"
        >
          <Settings className="h-5 w-5" />
          {allFieldsFilled && (
            <span className="absolute top-0 right-0 h-2.5 w-2.5 bg-green-500 rounded-full border-2 border-background" />
          )}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>AWS Credentials</DialogTitle>
          <DialogDescription>
            Configure your AWS environment variables. All fields are required.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="access-key-id">AWS Access Key ID</Label>
            <Input
              id="access-key-id"
              type="text"
              placeholder="Enter AWS Access Key ID"
              value={credentials.AWS_ACCESS_KEY_ID}
              onChange={(e) =>
                handleInputChange("AWS_ACCESS_KEY_ID", e.target.value)
              }
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="secret-access-key">AWS Secret Access Key</Label>
            <Input
              id="secret-access-key"
              type="password"
              placeholder="Enter AWS Secret Access Key"
              value={credentials.AWS_SECRET_ACCESS_KEY}
              onChange={(e) =>
                handleInputChange("AWS_SECRET_ACCESS_KEY", e.target.value)
              }
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="session-token">AWS Session Token</Label>
            <Input
              id="session-token"
              type="password"
              placeholder="Enter AWS Session Token"
              value={credentials.AWS_SESSION_TOKEN}
              onChange={(e) =>
                handleInputChange("AWS_SESSION_TOKEN", e.target.value)
              }
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="region">AWS Region</Label>
            <Input
              id="region"
              type="text"
              placeholder="e.g., us-east-1"
              value={credentials.AWS_REGION}
              onChange={(e) => handleInputChange("AWS_REGION", e.target.value)}
            />
          </div>
          {allFieldsFilled && (
            <div className="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800 rounded-md">
              <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400" />
              <span className="text-sm text-green-700 dark:text-green-300 font-medium">
                All credentials are configured and ready to use
              </span>
            </div>
          )}
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={!allFieldsFilled}>
            Save
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}


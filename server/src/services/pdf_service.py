import io
import base64
import re
import hashlib
import os
from io import BytesIO
import subprocess
import json
import tempfile
import uuid

# Try to import PyMuPDF, provide helpful error if it fails
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    print("WARNING: PyMuPDF (fitz) module not found. PDF processing will be limited.")
    print("Please install with: pip install PyMuPDF==1.24.1")
    PYMUPDF_AVAILABLE = False

# Try to import pdfplumber, another PDF extraction library
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    print("WARNING: pdfplumber module not found. PDF processing will be limited.")
    print("Please install with: pip install pdfplumber==0.10.3")
    PDFPLUMBER_AVAILABLE = False

# Try to import PyPDF2, a pure Python PDF library
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    print("WARNING: PyPDF2 module not found. PDF processing will be limited.")
    print("Please install with: pip install PyPDF2==3.0.1")
    PYPDF2_AVAILABLE = False

# Try to import OCR libraries
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    print("WARNING: OCR libraries (pdf2image, pytesseract) not found. PDF OCR will not be available.")
    print("Please install with: pip install pdf2image==1.17.0 pytesseract==0.3.10")
    OCR_AVAILABLE = False

def extract_with_client_style_method(pdf_bytes):
    """
    Extract text from PDF using a Node.js script that leverages PDF.js
    This mimics the client-side extraction approach for consistency
    
    Args:
        pdf_bytes: PDF content as bytes
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Save PDF to a temporary file
        temp_pdf_path = os.path.join(tempfile.gettempdir(), f"temp_pdf_{uuid.uuid4()}.pdf")
        temp_output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4()}.txt")
        
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_bytes)
        
        # Try using pdftotext utility first (poppler-utils) if available
        try:
            subprocess.run(["pdftotext", "-layout", temp_pdf_path, temp_output_path], 
                           check=True, 
                           stderr=subprocess.PIPE,
                           timeout=60)
            
            with open(temp_output_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                
            return text
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"pdftotext extraction failed: {str(e)}")
            
            # If pdftotext fails, try using pdf2text Python package
            if PYMUPDF_AVAILABLE:
                try:
                    doc = fitz.open(temp_pdf_path)
                    text = ""
                    
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        # Try different extraction methods for best results
                        try:
                            page_text = page.get_text("text", sort=True)
                            if not page_text.strip():
                                # If text method fails, try raw
                                page_text = page.get_text("raw")
                        except:
                            # Final fallback
                            page_text = page.get_text()
                            
                        text += page_text
                        if page_num < len(doc)-1:
                            text += f"\n\n--- Page {page_num+1} ---\n\n"
                    
                    doc.close()
                    return text
                except Exception as mupdf_error:
                    print(f"PyMuPDF extraction failed: {str(mupdf_error)}")
                    return ""
            else:
                return ""
    except Exception as e:
        print(f"Client-style PDF extraction error: {str(e)}")
        return ""
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except:
            pass

def extract_text_from_pdf_base64(pdf_base64):
    """
    Extract text from a base64-encoded PDF file using multiple methods with fallbacks
    
    Args:
        pdf_base64: Base64-encoded PDF content
        
    Returns:
        str: Extracted text from the PDF
    """
    if not any([PYMUPDF_AVAILABLE, PDFPLUMBER_AVAILABLE, PYPDF2_AVAILABLE]):
        return "PDF processing is not available because no PDF libraries are installed. Please install PyMuPDF, pdfplumber, or PyPDF2."
    
    try:
        # Check if the content is too small to be a valid PDF
        if len(pdf_base64) < 100:
            return "The provided content is too small to be a valid PDF file."
        
        # Prepare the PDF data
        pdf_bytes = None
        if pdf_base64.startswith("%PDF"):
            # Already raw PDF content, not base64
            pdf_bytes = pdf_base64.encode('utf-8')
        else:
            try:
                # Clean the base64 string (remove headers if present)
                base64_data = clean_base64_string(pdf_base64)
                # Decode base64 to bytes
                pdf_bytes = base64.b64decode(base64_data)
                
                # Verify it's actually a PDF
                if not pdf_bytes.startswith(b'%PDF'):
                    return "The provided content doesn't appear to be a valid PDF file. The content doesn't start with the PDF signature."
            except Exception as decode_error:
                return f"Failed to decode the provided content as a PDF: {str(decode_error)}"
        
        # Store the PDF bytes to a temporary file for libraries that require a file path
        temp_pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp", "temp.pdf")
        os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_bytes)
        
        # Try multiple extraction methods
        extraction_results = []
        extraction_errors = []
        metadata = {}

        # Method 0 (New): Client-style method (similar to PDF.js approach)
        try:
            client_style_text = extract_with_client_style_method(pdf_bytes)
            if client_style_text:
                extraction_results.append({"method": "Client-style extraction", "text": client_style_text, "score": len(client_style_text)})
        except Exception as e:
            extraction_errors.append(f"Client-style extraction error: {str(e)}")
        
        # Method 1: PyMuPDF (usually most reliable)
        if PYMUPDF_AVAILABLE:
            try:
                pymupdf_text, pymupdf_metadata, success = extract_with_pymupdf(pdf_bytes)
                if success and pymupdf_text:
                    extraction_results.append({"method": "PyMuPDF", "text": pymupdf_text, "score": len(pymupdf_text)})
                    metadata = pymupdf_metadata
                else:
                    extraction_errors.append(f"PyMuPDF extraction failed: {pymupdf_text}")
            except Exception as e:
                extraction_errors.append(f"PyMuPDF error: {str(e)}")
        
        # Method 2: pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                pdfplumber_text = extract_with_pdfplumber(pdf_bytes)
                if pdfplumber_text:
                    extraction_results.append({"method": "pdfplumber", "text": pdfplumber_text, "score": len(pdfplumber_text)})
            except Exception as e:
                extraction_errors.append(f"pdfplumber error: {str(e)}")
        
        # Method 3: PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                pypdf2_text = extract_with_pypdf2(pdf_bytes)
                if pypdf2_text:
                    extraction_results.append({"method": "PyPDF2", "text": pypdf2_text, "score": len(pypdf2_text)})
            except Exception as e:
                extraction_errors.append(f"PyPDF2 error: {str(e)}")
        
        # Method 4: OCR as a last resort
        if OCR_AVAILABLE and not extraction_results:
            try:
                ocr_text = extract_with_ocr(pdf_bytes)
                if ocr_text:
                    extraction_results.append({
                        "method": "OCR", 
                        "text": f"Note: This text was extracted using OCR and may contain errors.\n\n{ocr_text}", 
                        "score": len(ocr_text)
                    })
            except Exception as e:
                extraction_errors.append(f"OCR error: {str(e)}")
        
        # Clean up the temporary file
        try:
            os.remove(temp_pdf_path)
        except:
            pass
        
        # If we have extraction results, use the best one
        if extraction_results:
            # Sort by score (text length)
            extraction_results.sort(key=lambda x: x["score"], reverse=True)
            best_result = extraction_results[0]
            
            # Add a note about which method was used
            text = f"Extracted with {best_result['method']} (tried {len(extraction_results)} methods).\n\n{best_result['text']}"
            
            # Clean and normalize the text
            text = normalize_text(text)
            
            # Check if the extracted text contains mostly page breaks and little content
            content_lines = [line for line in text.split('\n') if line.strip() and "--- Page Break ---" not in line and "--- Page " not in line]
            if len(content_lines) < 5:
                if "LaTeX" in text:
                    text = f"The PDF appears to be a LaTeX template or document with minimal text content. Only {len(content_lines)} lines with actual content were extracted. This may be a template document rather than a complete paper.\n\n{text}"
                elif "<html" in text.lower() or "<body" in text.lower():
                    text = f"The PDF appears to contain HTML code rather than proper research paper content. Only {len(content_lines)} lines with actual content were extracted. This might be a web page converted to PDF rather than a research paper.\n\n{text}"
                else:
                    text = f"The PDF contains very little extractable text (only {len(content_lines)} lines with actual content). It might be a scan, an image-based PDF, or contain content that cannot be properly extracted.\n\n{text}"
            
            # Print statistics about the extraction
            print(f"Extracted {len(text)} characters of text from PDF")
            
            return text
        else:
            # If all methods failed, return an error with details
            detailed_error = "\n".join(extraction_errors)
            return f"Failed to extract text from the PDF using {len(extraction_errors)} different methods. Please try a different PDF file or manually copy the text.\n\nDetailed errors:\n{detailed_error}"
        
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"Error extracting text from PDF: {str(e)}"

def extract_with_pymupdf(pdf_bytes):
    """Extract text using PyMuPDF with detailed error handling"""
    try:
        # Open the PDF from bytes with error handling
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Check if PDF has pages
        if len(doc) == 0:
            doc.close()
            return "The PDF doesn't contain any pages.", {}, False
        
        text = ""
        html_content_detected = False
        page_content_lengths = []
        
        # Extract metadata
        metadata = doc.metadata
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Try different text extraction methods
            try:
                page_text = page.get_text()
                page_content_lengths.append(len(page_text.strip()))
                
                # Check for HTML content
                if "<html" in page_text.lower() or "<body" in page_text.lower() or "<div" in page_text.lower():
                    html_content_detected = True
                
                # If text extraction returns minimal content, try other methods
                if len(page_text.strip()) < 50:
                    try:
                        # Try extracting as raw (might help with some LaTeX PDFs)
                        raw_text = page.get_text("raw")
                        if len(raw_text.strip()) > len(page_text.strip()):
                            page_text = raw_text
                            page_content_lengths[-1] = len(raw_text.strip())
                            
                        # As a last resort, try HTML format
                        if len(page_text.strip()) < 50:
                            html_text = page.get_text("html")
                            if len(html_text.strip()) > len(page_text.strip()):
                                page_text = html_text
                                page_content_lengths[-1] = len(html_text.strip())
                                html_content_detected = True
                    except Exception as method_error:
                        print(f"Error when trying alternative extraction method: {str(method_error)}")
            except Exception as extract_error:
                print(f"Error extracting text from page {page_num}: {str(extract_error)}")
                page_text = f"[Error extracting text from page {page_num+1}: {str(extract_error)}]"
                page_content_lengths.append(0)
            
            text += page_text
            text += "\n\n--- Page Break ---\n\n"  # Add page break marker
        
        # Close the document
        doc.close()
        
        # Calculate statistics about the extracted content
        avg_content_per_page = sum(page_content_lengths) / len(page_content_lengths) if page_content_lengths else 0
        empty_pages = sum(1 for length in page_content_lengths if length < 50)
        
        # Check for potential issues with the extracted text
        if avg_content_per_page < 100 and len(page_content_lengths) > 1:
            text = f"Warning: This PDF contains very little text content (average {avg_content_per_page:.1f} characters per page, with {empty_pages} empty pages out of {len(page_content_lengths)} total). It may be a scan, image-based PDF, or template.\n\n{text}"
        
        if html_content_detected:
            text = f"Warning: This PDF contains HTML code. It might be a web page converted to PDF or have embedded HTML content.\n\n{text}"
            
        # Add page count to metadata
        metadata["page_count"] = len(page_content_lengths)
            
        return text, metadata, True
        
    except Exception as open_error:
        err_msg = str(open_error)
        if "zlib error" in err_msg:
            return "The PDF appears to be corrupted. There was a zlib decompression error which usually indicates the PDF is damaged or incomplete.", {}, False
        return f"Failed to open the PDF: {err_msg}", {}, False

def extract_with_pdfplumber(pdf_bytes):
    """Extract text using pdfplumber"""
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            # Extract text (or empty string if None)
            page_text = page.extract_text() or ""
            text += page_text
            text += "\n\n--- Page Break ---\n\n"  # Add page break marker
        return text

def extract_with_pypdf2(pdf_bytes):
    """Extract text using PyPDF2"""
    reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        # Extract text (or empty string if None)
        page_text = page.extract_text() or ""
        text += page_text
        text += "\n\n--- Page Break ---\n\n"  # Add page break marker
    return text

def extract_with_ocr(pdf_bytes):
    """Extract text using OCR (pdf2image + pytesseract)"""
    images = convert_from_bytes(pdf_bytes)
    text = ""
    for i, image in enumerate(images):
        page_text = pytesseract.image_to_string(image)
        text += page_text
        text += "\n\n--- Page Break ---\n\n"  # Add page break marker
    return text

def clean_base64_string(base64_str):
    """
    Clean a base64 string by removing headers and other non-base64 characters
    
    Args:
        base64_str: The base64 string to clean
        
    Returns:
        str: Cleaned base64 string
    """
    # Remove data URL header if present
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    
    # Remove any whitespace or newlines
    base64_str = re.sub(r'\s+', '', base64_str)
    
    # Ensure the string has valid base64 padding
    padding = 4 - (len(base64_str) % 4)
    if padding < 4:
        base64_str += '=' * padding
    
    return base64_str

def normalize_text(text):
    """
    Normalize text extracted from PDF
    
    Args:
        text: The text to normalize
        
    Returns:
        str: Normalized text
    """
    # Handle HTML content if present
    if re.search(r'<html|<body|<div|<span|<p\b|<a\b', text.lower()):
        # Check for repetitive HTML patterns
        html_blocks = re.findall(r'<html.*?</html>', text, re.DOTALL | re.IGNORECASE)
        
        # If we have multiple identical HTML blocks, keep only one
        if len(html_blocks) > 1:
            # Use hashing to check for duplicate content
            unique_blocks = {}
            for block in html_blocks:
                # Create a hash of the block to identify duplicates
                block_hash = hashlib.md5(block.encode()).hexdigest()
                unique_blocks[block_hash] = block
                
            if len(unique_blocks) < len(html_blocks):
                # Replace repetitive blocks with a note
                text = f"Note: The PDF contains {len(html_blocks)} HTML blocks, but only {len(unique_blocks)} unique blocks. Removing duplicates.\n\n"
                text += "\n\n".join(unique_blocks.values())
        
        # Simple HTML tag removal (a more robust solution would use an HTML parser)
        text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Convert HTML entities
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#\d+;', '', text)
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix hyphenated words that span lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Try to detect and clean up LaTeX commands and environments
    if '\\begin' in text or '\\end' in text or '\\' in text:
        # Replace common LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})+', ' ', text)
        # Clean up LaTeX environments
        text = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', ' ', text, flags=re.DOTALL)
        # Remove standalone LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    
    # Remove consecutive page breaks
    text = re.sub(r'(--- Page Break ---\s*)+', '--- Page Break ---\n\n', text)
    
    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    
    # Final cleanup of whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text

def extract_metadata_from_pdf(pdf_base64):
    """
    Extract metadata from a PDF file
    
    Args:
        pdf_base64: Base64-encoded PDF content
        
    Returns:
        dict: PDF metadata
    """
    if not PYMUPDF_AVAILABLE:
        return {"error": "PyMuPDF is not installed. PDF metadata extraction is not available."}
    
    try:
        # Clean the base64 string
        if pdf_base64.startswith("%PDF"):
            # Already raw PDF content, not base64
            pdf_bytes = pdf_base64.encode('utf-8')
        else:
            base64_data = clean_base64_string(pdf_base64)
            # Decode base64 to bytes
            pdf_bytes = base64.b64decode(base64_data)
        
        # Open the PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Extract metadata
        metadata = doc.metadata
        
        # Add page count
        metadata["page_count"] = len(doc)
        
        # Extract title from first page if not in metadata
        if not metadata.get("title"):
            try:
                first_page = doc.load_page(0)
                first_page_text = first_page.get_text()
                # Try to extract the first line as title
                title_match = re.search(r'^(.+?)[\n\r]', first_page_text.strip())
                if title_match:
                    metadata["title"] = title_match.group(1).strip()
            except:
                pass
        
        # Close the document
        doc.close()
        
        return metadata
    except Exception as e:
        print(f"Error extracting metadata from PDF: {str(e)}")
        return {"error": str(e)}

def is_pdf_content(content):
    """
    Check if the content is a PDF file
    
    Args:
        content: Content to check
        
    Returns:
        bool: True if content is a PDF, False otherwise
    """
    if not content:
        return False
        
    # Check if it starts with PDF header
    if content.startswith("%PDF"):
        return True
    
    # Check if it looks like base64-encoded PDF
    if "base64," in content and re.search(r'data:application/pdf', content):
        return True
    
    # Try to detect PDF in base64 format without header
    if len(content) > 100 and re.match(r'^[A-Za-z0-9+/]+={0,2}$', content[:100]):
        try:
            # Try to decode a small sample
            sample = base64.b64decode(clean_base64_string(content[:100]))
            # Check if it starts with PDF signature
            if sample.startswith(b'%PDF'):
                return True
        except:
            pass
    
    return False 
# Research Paper Assistant - Client

The client-side application for the Research Paper Assistant, built with React, TypeScript, and Vite.

## Features

- PDF viewer with text selection capabilities
- Interactive chat interface with real-time updates
- Paper content analysis and categorization
- Related research suggestions
- Responsive design with Tailwind CSS

## Project Structure

```
client/
├── public/              # Static assets
├── src/
│   ├── components/     # React components
│   │   ├── chat/       # Chat interface components
│   │   └── pdf/        # PDF viewer components
│   ├── hooks/          # Custom React hooks
│   ├── services/       # API and utility services
│   └── types/          # TypeScript type definitions
├── .env.example        # Environment variables template
└── package.json        # Project dependencies
```

## Development

### Prerequisites
- Node.js 16 or higher
- npm or yarn

### Setup
1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```

3. Update the `.env` file with your server URL:
   ```
   VITE_SERVER_URL=http://localhost:8000
   ```

### Running the Development Server
```bash
npm run dev
```

The application will be available at http://localhost:5173

### Building for Production
```bash
npm run build
```

The built files will be in the `dist` directory.

## Key Components

### Chat Interface
- Real-time message updates
- Support for text selection and PDF content
- Paper categorization display
- Error handling and loading states

### PDF Viewer
- PDF file upload and display
- Text selection and extraction
- Content analysis integration

## API Integration

The client communicates with the server through the following endpoints:
- `POST /api/queries/` - Submit a new query
- `GET /api/queries/` - Get query history
- `DELETE /api/queries/` - Clear query history

## Contributing

Please follow the project's coding standards and submit pull requests for any improvements.

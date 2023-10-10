import { useState } from 'react';
import IframeResizer from 'iframe-resizer-react';
import ExternalLink from '../components/ExternalLink';

type DemoProps = {
  id: string;
  src: string;
  height?: string;
  className?: string;
};

export function ExampleImages() {
  return (
    <span className="mb-[-24px] mr-4 w-full xl:w-4/5 text-sm text-right	opacity-80">
      example images:{' '}
      <ExternalLink href="https://huggingface.co/spaces/HuangLab/CELL-E_2-Image_Prediction/resolve/main/images/Armadillo%20repeat-containing%20X-linked%20protein%205%20nucleus.jpg">
        nucleus
      </ExternalLink>
      ,{' '}
      <ExternalLink href="https://huggingface.co/spaces/HuangLab/CELL-E_2-Image_Prediction/resolve/main/images/Armadillo%20repeat-containing%20X-linked%20protein%205%20protein.jpg">
        protein
      </ExternalLink>
    </span>
  );
}

function Demo({ id, src, height, className = '' }: DemoProps) {
  const [showDemo, setShowDemo] = useState(false);

  return (
    <div
      id={id}
      className="relative flex flex-col items-center mt-12 mb-16 px-2 xl:px-8 2xl:px-32"
    >
      <ExampleImages />
      {showDemo ? (
        <IframeResizer
          className={`rounded-lg w-full xl:w-4/5 ${className}`}
          src={src}
          frameBorder="0"
          scrolling={false} // == `scrolling="no"`
          style={{
            minWidth: '100%',
            minHeight: 1000
          }}
        />
      ) : (
        <button
          style={{
            height: height ? `${Number(height) / 3}px` : 400
          }}
          className={`
            flex justify-center items-center
            w-full xl:w-4/5
            bg-gray-100 text-indigo-500 hover:bg-gray-200
            font-semibold
            rounded-lg
            cursor-pointer
            transition
            border-[1px] border-dashed border-zinc-300 hover:border-gray-200
          `}
          onClick={() => setShowDemo(true)}
        >
          Click to load demo
        </button>
      )}
    </div>
  );
}

export default Demo;

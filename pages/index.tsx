import Image from 'next/image';
import Script from 'next/script';
import Head from 'next/head';
import { Inter } from 'next/font/google';
import Header from '../components/Header';
import Nav from '../components/Nav';
import Abstract from '../components/Abstract';
import Authors from '../components/Authors';
import Demo from '../components/Demo';
import Footer from '../components/Footer';

function Main() {
  return (
    <main
      role="main"
      className={`
        mx-auto px-4 xl:px-32
        min-h-screen max-w-container 
        bg-[#f6f9fc]
      `}
    >
      <div
        className={`
          relative mx-auto pt-24 
          max-w-[100rem] min-h-screen 
          border-l-[1px] border-r-[1px] border-dashed border-zinc-300 
        `}
      >
        <section
          className={`
            px-4 xl:px-32
            grid grid-cols-5 
            summary
          `}
          id="model"
        >
          <header className="text-[#0a2540] col-span-5 xl:col-span-2">
            <h3 className="text-indigo-500 font-bold">Section 1</h3>
            <h2 className="mt-6 mb-3 text-5xl font-bold tracking-tight">
              Model Overview
            </h2>
            <p>{''}</p>
          </header>
          <p
            className={`
              py-14 px-0 xl:px-6 
              text-[#425466] col-span-5 xl:col-span-3 text-lg
            `}
          >
            <a
              target="_blank"
              rel="noopener noreferrer"
              href="/CELL-E_2/architecture.png"
            >
              <Image
                className="xl:-mt-4 xl:-ml-4 mb-4"
                src="/CELL-E_2/architecture.png"
                alt="Model Overview"
                width="1000"
                height="600"
                priority
              />
            </a>
            CELL-E 2 is the second iteration of the original{' '}
            <a
              className="text-indigo-500 hover:text-zinc-900 font-bold"
              target="_blank"
              rel="noopener noreferrer"
              href="https://www.biorxiv.org/content/10.1101/2022.05.27.493774v1"
            >
              CELL-E
            </a>{' '}
            model which utilizes an amino acid sequence and nucleus image to
            make predictions of subcellular protein localization with respect to
            the nucleus. we use novel bidirectional transformer that can
            generate images depicting protein subcellular localization from the
            amino acid sequences (and vice versa). CELL-E 2 not only captures
            the spatial complexity of protein localization and produce
            probability estimates of localization atop a nucleus image, but also
            being able to generate sequences from images, enabling de novo
            protein design. We trained on the{' '}
            <a
              className="text-indigo-500 hover:text-zinc-900 font-bold"
              target="_blank"
              rel="noopener noreferrer"
              href="https://www.proteinatlas.org/"
            >
              Human Protein Atlas
            </a>{' '}
            (HPA) and the{' '}
            <a
              className="text-indigo-500 hover:text-zinc-900 font-bold"
              target="_blank"
              rel="noopener noreferrer"
              href="https://opencell.czbiohub.org/"
            >
              OpenCell
            </a>{' '}
            datasets. CELL-E 2 utilizes pretrained amino acid embeddings from{' '}
            <a
              className="text-indigo-500 hover:text-zinc-900 font-bold"
              target="_blank"
              rel="noopener noreferrer"
              href="https://github.com/facebookresearch/esm"
            >
              ESM-2
            </a>
            . Localization is predicted as a binary image atop the provided
            nucleus. The logit values are weighted against these binary images
            to produce a heatmap of expected localization.
          </p>
        </section>

        <section
          className={`
            px-4 xl:px-32
            grid grid-cols-5
          `}
        >
          <header className="text-[#0a2540] col-span-2">
            <h3 className="text-indigo-500 font-bold">Section 2</h3>
            <h2 className="mt-6 mb-3 text-5xl font-bold tracking-tight">
              Localization Prediction
            </h2>
            <p>Text-to-Image</p>
          </header>
          <p
            className={`
              py-14 px-0 xl:px-6 
              text-[#425466] col-span-5 xl:col-span-3 text-lg
            `}
          >
            {`CELL-E 2 can generate localization images by masking the image input section.`}
          </p>
        </section>

        <Demo
          id="demo1"
          src="https://huanglab-cell-e-2-image-prediction.hf.space/"
          height="1200"
        />

        <section
          className={`
            px-4 xl:px-32
            grid grid-cols-5
          `}
        >
          <header className="text-[#0a2540] col-span-2">
            <h3 className="text-indigo-500 font-bold">Section 3</h3>
            <h2 className="mt-6 mb-3 text-5xl font-bold tracking-tight">
              Sequence Prediction
            </h2>
            <p>Image-to-Text</p>
          </header>
          <p
            className={`
              py-14 px-0 xl:px-6 
              text-[#425466] col-span-5 xl:col-span-3 text-lg
            `}
          >
            {`Similarly, amino acids positions can be masked (replaced or inserted) to make predictions based on the localization pattern.`}
          </p>
        </section>

        <Demo
          id="demo2"
          src="https://huanglab-cell-e-2-sequence-prediction.hf.space/"
          height="1230"
        />

        <section
          className={`
            px-4 xl:px-32
            grid grid-cols-5
          `}
        >
          <header className="text-[#0a2540] col-span-2">
            <h3 className="text-indigo-500 font-bold">Section 4</h3>
            <h2 className="mt-6 mb-3 text-5xl font-bold tracking-tight">
              <em>De novo</em> Protein Design
            </h2>
            <p>{''}</p>
          </header>
          <p
            className={`
                py-14 px-0 xl:px-6 
                text-[#425466] col-span-5 xl:col-span-3 text-lg
              `}
          >
            {`We created an entirely new approach to protein design which leverages spatial information from images. Using CELL-E 2, we predicted 255 likely novel nuclear localizing signals with distinct sequence homology from documented sequences.`}
          </p>
        </section>
      </div>
    </main>
  );
}

function MetaTag() {
  return (
    <Head>
      <title>CELL-E 2</title>
      <meta
        name="description"
        content="CELL-E 2 is a bidirectional transformer that generates realistic images and sequences of protein localization in the cell."
        key="desc"
      />
      <meta property="og:title" content="CELL-E 2" />
      <meta
        property="og:description"
        content="CELL-E 2 is a bidirectional transformer that generates realistic images and sequences of protein localization in the cell."
      />
      <meta
        property="og:image"
        content="https://bohuanglab.github.io/CELL-E_2/architecture.png"
      />
      <meta property="og:image:alt" content="architecture figure" />
      <meta name="twitter:card" content="summary_large_image" />
    </Head>
  );
}

function GTag() {
  return (
    <>
      <Script src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID" />
      <Script id="google-analytics">
        {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
 
          gtag('config', 'GA_MEASUREMENT_ID');
        `}
      </Script>
    </>
  );
}

const inter = Inter({ subsets: ['latin'] });

export default function Home() {
  return (
    <div className={`relative ${inter.className}`}>
      <MetaTag />

      <Header />

      <Nav />

      <div className="mt-[-1px] border-t-[1px] border-solid border-zinc-800 bg-zinc-100">
        {/* top border */}
      </div>

      <Authors />

      <Abstract />

      <Main />

      <Footer />

      <GTag />
    </div>
  );
}

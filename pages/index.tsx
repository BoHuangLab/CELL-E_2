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
import Video from '../components/Video';
import ExternalLink from '../components/ExternalLink';
import React from 'react';

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
        <a
          className="flex justify-center"
          target="_blank"
          rel="noopener noreferrer"
          href="/CELL-E_2/architecture.png"
        >
          <Image
            className="xl:-mt-4 mb-4 transition ease-in-out hover:scale-110"
            src="/CELL-E_2/architecture.png"
            alt="Model Overview"
            width="960"
            height="600"
            priority
          />
        </a>
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
          <div
            className={`
              py-14 px-0 xl:px-6 
              text-[#425466] col-span-5 2xl:col-span-3 text-lg
            `}
          >
            <p className="mb-2">
              CELL-E 2 is the second iteration of the original{' '}
              <ExternalLink href="https://www.biorxiv.org/content/10.1101/2022.05.27.493774v1">
                CELL-E
              </ExternalLink>{' '}
              model which utilizes an amino acid sequence and nucleus image to
              make predictions of subcellular protein localization with respect
              to the nucleus.{' '}
            </p>
            <p className="mb-2">
              We use a novel bidirectional transformer that can generate images
              depicting protein subcellular localization from the amino acid
              sequences (and vice versa).{' '}
            </p>
            <p className="mb-2">
              CELL-E 2 can not only capture the spatial complexity of protein
              localization and produce probability estimates of localization
              atop a nucleus image, but it is also able to generate sequences
              from images, enabling de novo protein design.
            </p>
            <p className="mb-2">
              We trained on the{' '}
              <ExternalLink href="https://www.proteinatlas.org/">
                Human Protein Atlas
              </ExternalLink>{' '}
              (HPA) and the{' '}
              <ExternalLink href="https://opencell.czbiohub.org/">
                OpenCell
              </ExternalLink>{' '}
              datasets. CELL-E 2 utilizes pretrained amino acid embeddings from{' '}
              <ExternalLink href="https://github.com/facebookresearch/esm">
                ESM-2
              </ExternalLink>
              . Localization is predicted as a binary image atop the provided
              nucleus. The logit values are weighted against these binary images
              to produce a <em>heatmap of expected localization.</em>
            </p>
          </div>
        </section>

        <section
          className={`
            px-4 xl:px-32
            grid grid-cols-5
          `}
        >
          <header className="text-[#0a2540] col-span-5 xl:col-span-2">
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
          src="https://huanglab-cell-e-2-image-prediction.hf.space/?__theme=light"
        />

        <section
          className={`
            px-4 xl:px-32
            grid grid-cols-5
          `}
        >
          <header className="text-[#0a2540] col-span-5 xl:col-span-2">
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
          src="https://huanglab-cell-e-2-sequence-prediction.hf.space/?__theme=light"
        />

        <section
          id="design"
          className={`
            px-4 xl:px-32
            grid grid-cols-5
          `}
        >
          <header className="text-[#0a2540] col-span-5 xl:col-span-2">
            <h3 className="text-indigo-500 font-bold">Section 4</h3>
            <h2 className="mt-6 mb-3 text-5xl font-bold tracking-tight">
              <em>De novo</em> Protein Design
            </h2>
            <p>{''}</p>
          </header>
          <div
            className={`
              py-14 px-0 xl:px-6 
              text-[#425466] col-span-5 xl:col-span-3 text-lg
            `}
          >
            <p className="mb-2">{`We created an entirely new approach to protein design which leverages spatial information from images.`}</p>
            <p className="mb-2">{`Using CELL-E 2, we predicted 255 likely novel nuclear localizing signals with distinct sequence homology from documented sequences.`}</p>
          </div>
        </section>

        <section
          id="comparison"
          className={`
            px-4 xl:px-32
            grid grid-cols-5
          `}
        >
          <header className="text-[#0a2540] col-span-5 xl:col-span-2">
            <h3 className="text-indigo-500 font-bold">Section 5</h3>
            <h2 className="mt-6 mb-3 text-5xl font-bold tracking-tight">
              Comparison
            </h2>
            <p>{''}</p>
          </header>
          <div
            className={`
              py-14 px-0 xl:px-6 
              text-[#425466] col-span-5 xl:col-span-3 text-lg
            `}
          >
            <Video />
            <p className="mb-2">{`In comparison to CELL-E, CELL-E 2 makes image predictions 65x faster with higher accuracy.`}</p>
          </div>
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
      <meta property="og:image" content="https://i.imgur.com/rUq86ih.png" />
      <meta property="og:image:alt" content="CELL-E 2 preview" />
      <meta name="twitter:card" content="summary_large_image" />
    </Head>
  );
}

function GTag() {
  return (
    <>
      <Script src="https://www.googletagmanager.com/gtag/js?id=G-321VEL09L6" />
      <Script id="google-analytics">
        {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
 
          gtag('config', 'G-321VEL09L6');
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

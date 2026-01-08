import VesselMap from '@/components/Map';
import { mockVessels } from '@/data/mockVessels';

export default function Home() {
  return (
    <div className="flex flex-col h-screen">
      <header className="bg-gray-900 text-white p-4 shadow-lg">
        <h1 className="text-2xl font-bold">LNG Vessel Tracker</h1>
        <p className="text-sm text-gray-400">Tracking {mockVessels.length} vessels worldwide</p>
      </header>
      <main className="flex-1 relative">
        <VesselMap vessels={mockVessels} />
      </main>
    </div>
  );
}

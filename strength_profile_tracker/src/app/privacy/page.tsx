export default function PrivacyPolicy() {
  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-6">Privacy Policy</h1>
        <p className="text-sm text-gray-500 mb-6">Last updated: December 21, 2024</p>

        <div className="space-y-6 text-gray-700">
          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Overview</h2>
            <p>
              REPPIT is a strength training tracker app that respects your privacy.
              We believe your fitness data belongs to you alone.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Data Collection</h2>
            <p className="mb-2">
              <strong>We do not collect any personal data.</strong>
            </p>
            <p>
              All your workout data, profile information, and preferences are stored
              locally on your device using browser localStorage. This data never
              leaves your device and is not transmitted to any servers.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Data Stored Locally</h2>
            <p className="mb-2">The following data is stored only on your device:</p>
            <ul className="list-disc list-inside space-y-1 ml-2">
              <li>Profile information (name, age, height, weight)</li>
              <li>Workout history and exercise logs</li>
              <li>Strength level ratings</li>
              <li>App preferences (theme, units)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">No Account Required</h2>
            <p>
              REPPIT works without any account or login. You can use all features
              of the app without providing any personal information.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Third-Party Services</h2>
            <p>
              This app does not integrate with any third-party analytics, advertising,
              or tracking services.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Data Deletion</h2>
            <p>
              You can delete all your data at any time by clearing your browser data
              or uninstalling the app. Since no data is stored on external servers,
              deletion is immediate and complete.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Children&apos;s Privacy</h2>
            <p>
              This app is intended for users aged 13 and older. We do not knowingly
              collect any information from children.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Changes to This Policy</h2>
            <p>
              We may update this privacy policy from time to time. Any changes will
              be reflected on this page with an updated revision date.
            </p>
          </section>

          <section>
            <h2 className="text-lg font-semibold text-gray-900 mb-2">Contact</h2>
            <p>
              If you have any questions about this privacy policy, please contact us
              at: <a href="mailto:support@reppit.app" className="text-blue-600 hover:underline">support@reppit.app</a>
            </p>
          </section>
        </div>

        <div className="mt-8 pt-6 border-t border-gray-200 text-center text-sm text-gray-500">
          <p>REPPIT - Your Personal Strength Training Companion</p>
        </div>
      </div>
    </div>
  )
}
